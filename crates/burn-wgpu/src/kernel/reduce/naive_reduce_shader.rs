use std::marker::PhantomData;

use crate::{
    codegen::{
        dialect::gpu::{gpu, Elem, Item, Scope, Variable, Visibility},
        execute_dynamic, Compilation, CompilationInfo, CompilationSettings, Compiler, EagerHandle,
        InputInfo, OutputInfo, WorkgroupLaunch,
    },
    element::JitElement,
    kernel::{DynamicKernelSource, SourceTemplate},
    tensor::JitTensor,
    Runtime,
};

pub(crate) trait NaiveReduceDim: Send + Sync + 'static {
    type Accumulator: Copy;

    fn initialize(scope: &mut Scope, input_item: Item, output_item: Item) -> Self::Accumulator;

    fn inner_loop(
        scope: &mut Scope,
        accumulator: Self::Accumulator,
        current_value: Variable,
        i: Variable,
    );

    fn assign(
        scope: &mut Scope,
        output: Variable,
        accumulator: Self::Accumulator,
        shape_reduce_dim: Variable,
    );
}

pub(crate) struct SumDim;

impl NaiveReduceDim for SumDim {
    type Accumulator = Variable;

    fn initialize(scope: &mut Scope, _input_item: Item, output_item: Item) -> Variable {
        scope.zero(output_item)
    }

    fn inner_loop(scope: &mut Scope, accumulator: Variable, value: Variable, _i: Variable) {
        gpu!(scope, accumulator += value);
    }

    fn assign(
        scope: &mut Scope,
        output: Variable,
        accumulator: Variable,
        _shape_reduce_dim: Variable,
    ) {
        let id = Variable::Id;
        gpu!(scope, output[id] = accumulator);
    }
}

pub(crate) struct MeanDim;

impl NaiveReduceDim for MeanDim {
    type Accumulator = Variable;

    fn initialize(scope: &mut Scope, _input_item: Item, output_item: Item) -> Variable {
        scope.zero(output_item)
    }

    fn inner_loop(scope: &mut Scope, accumulator: Variable, value: Variable, _i: Variable) {
        gpu!(scope, accumulator += value);
    }

    fn assign(
        scope: &mut Scope,
        output: Variable,
        accumulator: Variable,
        shape_reduce_dim: Variable,
    ) {
        let id = Variable::Id;
        let denominator = scope.create_local(accumulator.item());
        gpu!(scope, denominator = cast(shape_reduce_dim));
        gpu!(scope, accumulator = accumulator / denominator);
        gpu!(scope, output[id] = accumulator);
    }
}

pub(crate) struct ArgMax;

impl NaiveReduceDim for ArgMax {
    type Accumulator = (Variable, Variable);

    fn initialize(scope: &mut Scope, input_item: Item, _output_item: Item) -> Self::Accumulator {
        let max = scope.create_local(input_item);
        let index = scope.create_local(Elem::UInt);
        gpu!(scope, max = cast(-32767.0));
        (max, index)
    }

    fn inner_loop(
        scope: &mut Scope,
        (max, index): Self::Accumulator,
        value: Variable,
        i: Variable,
    ) {
        let condition = scope.create_local(Elem::Bool);
        gpu!(scope, condition = value > max);
        gpu!(scope, if(condition).then(|scope| {
            gpu!(scope, max = value);
            gpu!(scope, index = i);
        }));
    }

    fn assign(
        scope: &mut Scope,
        output: Variable,
        (_max, index): Self::Accumulator,
        _shape_reduce_dim: Variable,
    ) {
        let id = Variable::Id;
        gpu!(scope, output[id] = index);
    }
}

pub(crate) struct ArgMin;

impl NaiveReduceDim for ArgMin {
    type Accumulator = (Variable, Variable);

    fn initialize(scope: &mut Scope, input_item: Item, _output_item: Item) -> Self::Accumulator {
        let min = scope.create_local(input_item);
        let index = scope.create_local(Elem::UInt);
        gpu!(scope, min = cast(32767.0));
        (min, index)
    }

    fn inner_loop(
        scope: &mut Scope,
        (min, index): Self::Accumulator,
        value: Variable,
        i: Variable,
    ) {
        let condition = scope.create_local(Elem::Bool);
        gpu!(scope, condition = value < min);
        gpu!(scope, if(condition).then(|scope| {
            gpu!(scope, min = value);
            gpu!(scope, index = i);
        }));
    }

    fn assign(
        scope: &mut Scope,
        output: Variable,
        (_min, index): Self::Accumulator,
        _shape_reduce_dim: Variable,
    ) {
        let id = Variable::Id;
        gpu!(scope, output[id] = index);
    }
}

pub(crate) struct NaiveReduceDimComputeShader<RD: NaiveReduceDim> {
    tensor: Variable,
    dim: usize,
    output: Variable,
    _reduce_dim: PhantomData<RD>,
}

#[derive(new)]
pub(crate) struct NaiveReduceDimEagerKernel<
    RD: NaiveReduceDim,
    R: Runtime,
    EI: JitElement,
    EO: JitElement,
> {
    dim: usize,
    reduce_dim: PhantomData<RD>,
    _runtime: PhantomData<R>,
    _elem_in: PhantomData<EI>,
    _elem_out: PhantomData<EO>,
}

impl<RD: NaiveReduceDim, R: Runtime, EI: JitElement, EO: JitElement> DynamicKernelSource
    for NaiveReduceDimEagerKernel<RD, R, EI, EO>
{
    fn source(&self) -> crate::kernel::SourceTemplate {
        let mut scope = Scope::root();
        let item_input = EI::gpu_elem().into();
        let item_output = EO::gpu_elem().into();

        let tensor = Variable::GlobalInputArray(0, item_input);
        let output = Variable::GlobalOutputArray(0, item_output);

        NaiveReduceDimComputeShader {
            tensor,
            dim: self.dim,
            output,
            _reduce_dim: PhantomData::<RD>,
        }
        .expand(&mut scope);

        scope.write_global_custom(output);

        let tensor = InputInfo::Array {
            item: item_input,
            visibility: Visibility::Read,
        };

        let out = OutputInfo::Array { item: item_output };

        let info = CompilationInfo {
            inputs: vec![tensor],
            outputs: vec![out],
            scope,
        };

        let settings = CompilationSettings::default();
        let shader = Compilation::new(info).compile(settings);
        let shader = <R::Compiler as Compiler>::compile(shader);
        SourceTemplate::new(shader.to_string())
    }

    fn id(&self) -> String {
        format!("{:?}dim={}", core::any::TypeId::of::<Self>(), self.dim)
    }
}

impl<RD: NaiveReduceDim> NaiveReduceDimComputeShader<RD> {
    pub(crate) fn expand(self, scope: &mut Scope) {
        let tensor = self.tensor;
        let dim: Variable = self.dim.into();
        let id = Variable::Id;
        let output = self.output;

        let offset_input = scope.zero(Elem::UInt);
        let stride_input_dim = scope.create_local(Elem::UInt);
        let shape_input_dim = scope.create_local(Elem::UInt);

        gpu!(
            scope,
            range(0u32, Variable::Rank).for_each(|i, scope| {
                let stride_input = scope.create_local(Elem::UInt);
                let stride_output = scope.create_local(Elem::UInt);
                let shape_output = scope.create_local(Elem::UInt);

                gpu!(scope, stride_input = stride(tensor, i));
                gpu!(scope, stride_output = stride(output, i));
                gpu!(scope, shape_output = shape(output, i));

                let offset_local = scope.create_local(Elem::UInt);
                gpu!(scope, offset_local = id / stride_output);
                gpu!(scope, offset_local = offset_local % shape_output);

                let is_dim_reduce = scope.create_local(Elem::Bool);
                gpu!(scope, is_dim_reduce = i == dim);

                gpu!(scope, if(is_dim_reduce).then(|scope|{
                    gpu!(scope, shape_input_dim = shape(tensor, i));
                    gpu!(scope, stride_input_dim = stride_input);
                    gpu!(scope, offset_input += offset_local);
                }).else(|scope|{
                    gpu!(scope, offset_local = offset_local * stride_input);
                    gpu!(scope, offset_input += offset_local);
                }));
            })
        );

        let accumulator = RD::initialize(scope, tensor.item(), output.item());

        gpu!(
            scope,
            range(0u32, shape_input_dim).for_each(|i, scope| {
                let index = scope.create_local(Elem::UInt);
                gpu!(scope, index = i * stride_input_dim);
                gpu!(scope, index += offset_input);
                let value = scope.create_local(tensor.item());
                gpu!(scope, value = tensor[index]);
                RD::inner_loop(scope, accumulator, value, i);
            })
        );

        RD::assign(scope, output, accumulator, shape_input_dim);
    }
}

pub(crate) fn reduce_dim_naive<
    RD: NaiveReduceDim,
    R: Runtime,
    EI: JitElement,
    EO: JitElement,
    const D: usize,
>(
    input: JitTensor<R, EI, D>,
    output: JitTensor<R, EO, D>,
    dim: usize,
) -> JitTensor<R, EO, D> {
    let kernel = NaiveReduceDimEagerKernel::new(dim);

    execute_dynamic::<R, NaiveReduceDimEagerKernel<RD, R, EI, EO>, EI>(
        &[EagerHandle::new(
            &input.handle,
            &input.strides,
            &input.shape.dims,
        )],
        &[EagerHandle::new(
            &output.handle,
            &output.strides,
            &output.shape.dims,
        )],
        None,
        kernel,
        WorkgroupLaunch::Output { pos: 0 },
        input.client,
    );

    output
}

/// Execute the sum dim kernel.
pub fn sum_dim_naive<R: Runtime, E: JitElement, const D: usize>(
    input: JitTensor<R, E, D>,
    output: JitTensor<R, E, D>,
    dim: usize,
) -> JitTensor<R, E, D> {
    reduce_dim_naive::<SumDim, R, E, E, D>(input, output, dim)
}

/// Execute the mean dim kernel.
pub fn mean_dim_naive<R: Runtime, E: JitElement, const D: usize>(
    input: JitTensor<R, E, D>,
    output: JitTensor<R, E, D>,
    dim: usize,
) -> JitTensor<R, E, D> {
    reduce_dim_naive::<MeanDim, R, E, E, D>(input, output, dim)
}

/// Execute the argmax kernel.
pub fn argmax_naive<R: Runtime, E: JitElement, I: JitElement, const D: usize>(
    input: JitTensor<R, E, D>,
    dim: usize,
) -> JitTensor<R, I, D> {
    let mut shape_out = input.shape.clone();
    shape_out.dims[dim] = 1;
    let num_elems = shape_out.num_elements();
    let buffer = input.client.empty(num_elems * core::mem::size_of::<I>());
    let output = JitTensor::new(
        input.client.clone(),
        input.device.clone(),
        shape_out,
        buffer,
    );
    reduce_dim_naive::<ArgMax, R, E, I, D>(input, output, dim)
}

/// Execute the argmin kernel.
pub fn argmin_naive<R: Runtime, E: JitElement, I: JitElement, const D: usize>(
    input: JitTensor<R, E, D>,
    dim: usize,
) -> JitTensor<R, I, D> {
    let mut shape_out = input.shape.clone();
    shape_out.dims[dim] = 1;
    let num_elems = shape_out.num_elements();
    let buffer = input.client.empty(num_elems * core::mem::size_of::<I>());
    let output = JitTensor::new(
        input.client.clone(),
        input.device.clone(),
        shape_out,
        buffer,
    );
    reduce_dim_naive::<ArgMin, R, E, I, D>(input, output, dim)
}

#[cfg(test)]
mod tests {
    use crate::{
        kernel::reduce::{argmax_naive, init_reduce_output, mean_dim_naive},
        tests::{ReferenceBackend, TestBackend},
    };
    use burn_tensor::{ops::IntTensorOps, Data, Distribution, Int, Shape, Tensor};

    use super::sum_dim_naive;

    #[test]
    fn reduction_sum_dim_should_work_with_multiple_invocations() {
        let tensor =
            Tensor::<TestBackend, 2>::random([6, 1024], Distribution::Default, &Default::default());
        let tensor_ref =
            Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data(), &Default::default());
        let reduce_dim = 1;
        let output = init_reduce_output(&tensor.clone().into_primitive(), reduce_dim);

        let val = Tensor::<TestBackend, 2>::from_primitive(sum_dim_naive(
            tensor.into_primitive(),
            output,
            reduce_dim,
        ));
        let val_ref = tensor_ref.sum_dim(1);

        val_ref.into_data().assert_approx_eq(&val.into_data(), 3);
    }

    #[test]
    fn reduction_args_dim_should_work_with_multiple_invocations() {
        let tensor =
            Tensor::<TestBackend, 2>::random([6, 1024], Distribution::Default, &Default::default());
        let tensor_ref =
            Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data(), &Default::default());

        let val =
            Tensor::<TestBackend, 2, Int>::from_primitive(argmax_naive(tensor.into_primitive(), 1));
        let val_ref = tensor_ref.argmax(1);

        assert_eq!(val_ref.into_data().convert(), val.into_data());
    }

    #[test]
    fn sum_dim_should_work_with_int() {
        let summed_shape = Shape::new([1]);
        let data = Data::from([1, 2, 3, 4]);
        let tensor = TestBackend::int_from_data(data, &Default::default());

        let summed_tensor = TestBackend::int_empty(summed_shape, &Default::default());

        let val =
            Tensor::<TestBackend, 1, Int>::from_primitive(sum_dim_naive(tensor, summed_tensor, 0));

        let sum_as_data = Data::from([10]);
        val.into_data().assert_approx_eq(&sum_as_data, 1);
    }

    #[test]
    fn mean_dim_should_work_with_int() {
        let mean_shape = Shape::new([1]);
        let data = Data::from([1, 2, 3, 4]);
        let tensor = TestBackend::int_from_data(data, &Default::default());

        let mean_tensor = TestBackend::int_empty(mean_shape, &Default::default());

        let val =
            Tensor::<TestBackend, 1, Int>::from_primitive(mean_dim_naive(tensor, mean_tensor, 0));

        // Mean calculation truncates to an integer
        let mean_as_data = Data::from([2]);
        val.into_data().assert_approx_eq(&mean_as_data, 1);
    }
}
