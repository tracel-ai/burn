use std::marker::PhantomData;

use crate::{
    codegen::{
        dialect::gpu::{gpu, Elem, Scope, Variable, Visibility},
        execute_dynamic, Compilation, CompilationInfo, CompilationSettings, Compiler, EagerHandle,
        InputInfo, OutputInfo, WorkgroupLaunch,
    },
    element::JitElement,
    kernel::{DynamicKernelSource, SourceTemplate},
    tensor::JitTensor,
    Runtime,
};

use super::ReduceDimAlgorithm;

pub(crate) struct NaiveReduceDimComputeShader<RD: ReduceDimAlgorithm> {
    tensor: Variable,
    dim: usize,
    output: Variable,
    _reduce_dim: PhantomData<RD>,
}

#[derive(new)]
pub(crate) struct NaiveReduceDimEagerKernel<
    RD: ReduceDimAlgorithm,
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

impl<RD: ReduceDimAlgorithm, R: Runtime, EI: JitElement, EO: JitElement> DynamicKernelSource
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

impl<RD: ReduceDimAlgorithm> NaiveReduceDimComputeShader<RD> {
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

        let accumulator = RD::initialize_naive(scope, tensor.item(), output.item());

        gpu!(
            scope,
            range(0u32, shape_input_dim).for_each(|i, scope| {
                let index = scope.create_local(Elem::UInt);
                gpu!(scope, index = i * stride_input_dim);
                gpu!(scope, index += offset_input);
                let value = scope.create_local(tensor.item());
                gpu!(scope, value = tensor[index]);
                RD::inner_loop_naive(scope, accumulator, value, i);
            })
        );

        RD::assign_naive(scope, output, accumulator, shape_input_dim);
    }
}

/// Executes the naive kernel for reduce dim
pub fn reduce_dim_naive<
    RD: ReduceDimAlgorithm,
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

#[cfg(feature = "export_tests")]
mod tests {
    #[burn_tensor_testgen::testgen(reduction)]
    mod reduction {
        use super::*;
        use burn_tensor::{ops::IntTensorOps, Data, Distribution, Int, Shape, Tensor};
        use burn_jit::kernel::reduce::{sum_dim, ReduceStrategy};

        #[test]
        fn reduction_sum_dim_should_work_with_multiple_invocations() {
            let tensor = Tensor::<TestBackend, 2>::random(
                [6, 1024],
                Distribution::Default,
                &Default::default(),
            );
            let tensor_ref =
                Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data(), &Default::default());
            let reduce_dim = 1;

            let val =
                Tensor::<TestBackend, 2>::from_primitive(sum_dim::<TestRuntime, f32, f32, 2>(
                    tensor.into_primitive(),
                    reduce_dim,
                    ReduceStrategy::Naive,
                ));
            let val_ref = tensor_ref.sum_dim(1);

            val_ref.into_data().assert_approx_eq(&val.into_data(), 3);
        }

        // #[test]
        // fn reduction_argmin_dim_should_work_with_multiple_invocations() {
        //     let tensor =
        //         Tensor::<TestBackend, 2>::random([6, 1024], Distribution::Default, &Default::default());
        //     let tensor_ref =
        //         Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data(), &Default::default());
        //     let reduce_dim = 1;
        //     let output = init_reduce_output(&tensor.clone().into_primitive(), reduce_dim);

        //     let val = Tensor::<TestBackend, 2, Int>::from_primitive(reduce_dim_naive::<
        //         ArgMin,
        //         TestRuntime,
        //         f32,
        //         i32,
        //         2,
        //     >(
        //         tensor.into_primitive(),
        //         output,
        //         reduce_dim,
        //     ));
        //     let val_ref = tensor_ref.argmin(reduce_dim);

        //     assert_eq!(val_ref.into_data().convert(), val.into_data());
        // }

        // #[test]
        // fn reduction_argmax_dim_should_work_with_multiple_invocations() {
        //     let tensor =
        //         Tensor::<TestBackend, 2>::random([6, 1024], Distribution::Default, &Default::default());
        //     let tensor_ref =
        //         Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data(), &Default::default());
        //     let reduce_dim = 1;
        //     let output = init_reduce_output(&tensor.clone().into_primitive(), reduce_dim);

        //     let val = Tensor::<TestBackend, 2, Int>::from_primitive(reduce_dim_naive::<
        //         ArgMax,
        //         TestRuntime,
        //         f32,
        //         i32,
        //         2,
        //     >(
        //         tensor.into_primitive(),
        //         output,
        //         reduce_dim,
        //     ));
        //     let val_ref = tensor_ref.argmax(reduce_dim);

        //     assert_eq!(val_ref.into_data().convert(), val.into_data());
        // }

        // #[test]
        // fn sum_dim_should_work_with_int() {
        //     let summed_shape = Shape::new([1]);
        //     let data = Data::from([1, 2, 3, 4]);
        //     let tensor = TestBackend::int_from_data(data, &Default::default());

        //     let summed_tensor = TestBackend::int_empty(summed_shape, &Default::default());

        //     let val = Tensor::<TestBackend, 1, Int>::from_primitive(reduce_dim_naive::<
        //         SumDim,
        //         TestRuntime,
        //         i32,
        //         i32,
        //         1,
        //     >(tensor, summed_tensor, 0));

        //     let sum_as_data = Data::from([10]);
        //     val.into_data().assert_approx_eq(&sum_as_data, 1);
        // }

        // #[test]
        // fn mean_dim_should_work_with_int() {
        //     let mean_shape = Shape::new([1]);
        //     let data = Data::from([1, 2, 3, 4]);
        //     let tensor = TestBackend::int_from_data(data, &Default::default());

        //     let mean_tensor = TestBackend::int_empty(mean_shape, &Default::default());

        //     let val = Tensor::<TestBackend, 1, Int>::from_primitive(reduce_dim_naive::<
        //         MeanDim,
        //         TestRuntime,
        //         i32,
        //         i32,
        //         1,
        //     >(tensor, mean_tensor, 0));

        //     // Mean calculation truncates to an integer
        //     let mean_as_data = Data::from([2]);
        //     val.into_data().assert_approx_eq(&mean_as_data, 1);
        // }
    }
}
