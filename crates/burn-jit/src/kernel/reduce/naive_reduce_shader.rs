use std::marker::PhantomData;

use crate::{
    codegen::{
        dialect::gpu::{gpu, Elem, Scope, Variable, Visibility},
        Compilation, CompilationInfo, CompilationSettings, EagerHandle, Execution, InputInfo,
        OutputInfo, WorkgroupLaunch,
    },
    element::JitElement,
    gpu::ComputeShader,
    kernel::GpuComputeShaderPhase,
    tensor::JitTensor,
    Runtime,
};

use super::ReduceDimAlgorithm;

pub(crate) struct NaiveReduceDimComputeShader<E: JitElement, RD: ReduceDimAlgorithm<E>> {
    tensor: Variable,
    dim: usize,
    output: Variable,
    _reduce_dim: PhantomData<RD>,
    _elem: PhantomData<E>,
}

#[derive(new)]
pub(crate) struct NaiveReduceDimEagerKernel<
    RD: ReduceDimAlgorithm<EI>,
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

impl<RD: ReduceDimAlgorithm<EI>, R: Runtime, EI: JitElement, EO: JitElement> GpuComputeShaderPhase
    for NaiveReduceDimEagerKernel<RD, R, EI, EO>
{
    fn compile(&self) -> ComputeShader {
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
            _elem: PhantomData::<EI>,
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
        Compilation::new(info).compile(settings)
    }

    fn id(&self) -> String {
        format!("{:?}dim={}", core::any::TypeId::of::<Self>(), self.dim)
    }
}

impl<E: JitElement, RD: ReduceDimAlgorithm<E>> NaiveReduceDimComputeShader<E, RD> {
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
    let kernel = NaiveReduceDimEagerKernel::<RD, R, EI, EO>::new(dim);

    Execution::start(kernel, input.client)
        .inputs(&[EagerHandle::<R>::new(
            &input.handle,
            &input.strides,
            &input.shape.dims,
        )])
        .outputs(&[EagerHandle::new(
            &output.handle,
            &output.strides,
            &output.shape.dims,
        )])
        .execute(WorkgroupLaunch::Output { pos: 0 });

    output
}
