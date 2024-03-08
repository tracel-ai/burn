use std::marker::PhantomData;

use crate::{
    codegen::{Compilation, CompilationInfo, CompilationSettings, InputInfo, OutputInfo},
    gpu::{Scope, Variable, Visibility},
    kernel::{DynamicKernelSource, SourceTemplate},
    tensor::JitTensor,
    Compiler, JitElement, Runtime,
};

pub(crate) trait MaskStrategy {
    type Value;

    fn assign();
}

pub(crate) struct MaskFill<E> {
    _elem: PhantomData<E>,
}
impl<E: JitElement> MaskStrategy for MaskFill<E> {
    type Value = E;

    fn assign() {
        todo!()
    }
}

pub(crate) struct MaskWhere<R, E, const D: usize> {
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

impl<R: Runtime, E: JitElement, const D: usize> MaskStrategy for MaskWhere<R, E, D> {
    type Value = JitTensor<R, E, D>;

    fn assign() {
        todo!()
    }
}

pub(crate) struct MaskShader<EI: JitElement, EM: JitElement, M: MaskStrategy> {
    input: Variable,
    mask: Variable,
    output: Variable,
    _mask_strategy: PhantomData<M>,
    _input_elem: PhantomData<EI>,
    _mask_elem: PhantomData<EM>,
}

#[derive(new)]
pub(crate) struct MaskReadOnlyEagerKernel<M: MaskStrategy, R: Runtime, E: JitElement> {
    _mask: PhantomData<M>,
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

impl<M: MaskStrategy, R: Runtime, EI: JitElement, EM: JitElement> DynamicKernelSource
    for MaskReadOnlyEagerKernel<M, R, E>
{
    fn source(&self) -> crate::kernel::SourceTemplate {
        let mut scope = Scope::root();
        let tensor_item = EI::gpu_elem().into();
        let mask_item = EM::gpu_elem().into();

        let input = Variable::GlobalInputArray(0, tensor_item);
        let mask = Variable::GlobalInputArray(0, mask_item);
        let output = Variable::GlobalOutputArray(0, tensor_item);

        MaskShader {
            input,
            mask,
            output,
            _mask_strategy: PhantomData::<M>,
        }
        .expand(&mut scope);

        scope.write_global_custom(output);

        let input = InputInfo::Array {
            item: tensor_item,
            visibility: Visibility::Read,
        };

        let mask = InputInfo::Array {
            item: mask_item,
            visibility: Visibility::Read,
        };

        let out = OutputInfo::Array { item: tensor_item };

        let info = CompilationInfo {
            inputs: vec![input, mask],
            outputs: vec![out],
            scope,
        };

        let settings = CompilationSettings::default();
        let shader = Compilation::new(info).compile(settings);
        let shader = <R::Compiler as Compiler>::compile(shader);
        SourceTemplate::new(shader.to_string())
    }

    fn id(&self) -> String {
        format!("{:?}", core::any::TypeId::of::<Self>())
    }
}

impl<EI: JitElement, EM: JitElement, M: MaskStrategy> MaskShader<EI, EM, M> {
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
    RD: ReduceDimAlgorithm<EI>,
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
