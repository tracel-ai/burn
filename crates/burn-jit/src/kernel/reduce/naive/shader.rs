use burn_cube::{
    cpa,
    dialect::{ComputeShader, Elem, Scope, Variable, Visibility},
    Compilation, CompilationInfo, CompilationSettings, Execution, InputInfo, OutputInfo,
    TensorHandle, WorkgroupLaunch,
};
use std::marker::PhantomData;

use crate::{element::JitElement, kernel::GpuComputeShaderPhase, tensor::JitTensor, JitRuntime};

use super::base::ReduceDimNaive;

pub(crate) struct NaiveReduceDimComputeShader<E: JitElement, RD: ReduceDimNaive<E>> {
    tensor: Variable,
    dim: usize,
    output: Variable,
    _reduce_dim: PhantomData<RD>,
    _elem: PhantomData<E>,
}

#[derive(new)]
pub(crate) struct NaiveReduceDimEagerKernel<
    RD: ReduceDimNaive<EI>,
    R: JitRuntime,
    EI: JitElement,
    EO: JitElement,
> {
    dim: usize,
    reduce_dim: PhantomData<RD>,
    _runtime: PhantomData<R>,
    _elem_in: PhantomData<EI>,
    _elem_out: PhantomData<EO>,
}

impl<RD: ReduceDimNaive<EI>, R: JitRuntime, EI: JitElement, EO: JitElement> GpuComputeShaderPhase
    for NaiveReduceDimEagerKernel<RD, R, EI, EO>
{
    fn compile(&self) -> ComputeShader {
        let mut scope = Scope::root();
        let item_input = EI::cube_elem().into();
        let item_output = EO::cube_elem().into();

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

impl<E: JitElement, RD: ReduceDimNaive<E>> NaiveReduceDimComputeShader<E, RD> {
    pub(crate) fn expand(self, scope: &mut Scope) {
        let tensor = self.tensor;
        let dim: Variable = self.dim.into();
        let id = Variable::Id;
        let output = self.output;

        let offset_input = scope.zero(Elem::UInt);
        let stride_input_dim = scope.create_local(Elem::UInt);
        let shape_input_dim = scope.create_local(Elem::UInt);

        cpa!(
            scope,
            range(0u32, Variable::Rank).for_each(|i, scope| {
                let stride_input = scope.create_local(Elem::UInt);
                let stride_output = scope.create_local(Elem::UInt);
                let shape_output = scope.create_local(Elem::UInt);

                cpa!(scope, stride_input = stride(tensor, i));
                cpa!(scope, stride_output = stride(output, i));
                cpa!(scope, shape_output = shape(output, i));

                let offset_local = scope.create_local(Elem::UInt);
                cpa!(scope, offset_local = id / stride_output);
                cpa!(scope, offset_local = offset_local % shape_output);

                let is_dim_reduce = scope.create_local(Elem::Bool);
                cpa!(scope, is_dim_reduce = i == dim);

                cpa!(scope, if(is_dim_reduce).then(|scope|{
                    cpa!(scope, shape_input_dim = shape(tensor, i));
                    cpa!(scope, stride_input_dim = stride_input);
                    cpa!(scope, offset_input += offset_local);
                }).else(|scope|{
                    cpa!(scope, offset_local = offset_local * stride_input);
                    cpa!(scope, offset_input += offset_local);
                }));
            })
        );

        let accumulator = RD::initialize_naive(scope, tensor.item(), output.item());

        cpa!(
            scope,
            range(0u32, shape_input_dim).for_each(|i, scope| {
                let index = scope.create_local(Elem::UInt);
                cpa!(scope, index = i * stride_input_dim);
                cpa!(scope, index += offset_input);
                let value = scope.create_local(tensor.item());
                cpa!(scope, value = tensor[index]);
                RD::inner_loop_naive(scope, accumulator, value, i);
            })
        );

        RD::assign_naive(scope, output, accumulator, shape_input_dim);
    }
}

/// Executes the naive kernel for reduce dim
pub fn reduce_dim_naive<
    RD: ReduceDimNaive<EI>,
    R: JitRuntime,
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
        .inputs(&[TensorHandle::<R>::new(
            &input.handle,
            &input.strides,
            &input.shape.dims,
        )])
        .outputs(&[TensorHandle::new(
            &output.handle,
            &output.strides,
            &output.shape.dims,
        )])
        .execute(WorkgroupLaunch::Output { pos: 0 });

    output
}
