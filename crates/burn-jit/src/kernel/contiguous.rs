use std::marker::PhantomData;

use crate::{
    codegen::{
        Compilation, CompilationInfo, CompilationSettings, EagerHandle, Execution, InputInfo,
        OutputInfo, WorkgroupLaunch,
    },
    gpu::{gpu, ComputeShader, Elem, IndexOffsetGlobalWithLayout, Scope, Variable, Visibility},
    tensor::JitTensor,
    JitElement, Runtime,
};

use super::GpuComputeShaderPhase;

pub(crate) struct IntoContiguousShader {
    tensor: Variable,
    output: Variable,
}

#[derive(new)]
pub(crate) struct IntoContiguousEagerKernel<R: Runtime, E: JitElement> {
    _runtime: PhantomData<R>,
    _elem_out: PhantomData<E>,
}

/// Make a jit tensor contiguous.
pub fn into_contiguous<R: Runtime, E: JitElement, const D: usize>(
    tensor: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    if tensor.is_contiguous() {
        return tensor;
    }

    let kernel = IntoContiguousEagerKernel::<R, E>::new();
    let num_elems = tensor.shape.num_elements();
    let buffer = tensor.client.empty(num_elems * core::mem::size_of::<E>());
    let output = JitTensor::new(
        tensor.client.clone(),
        tensor.device,
        tensor.shape.clone(),
        buffer,
    );

    Execution::start(kernel, tensor.client)
        .inputs(&[EagerHandle::<R>::new(
            &tensor.handle,
            &tensor.strides,
            &tensor.shape.dims,
        )])
        .outputs(&[EagerHandle::new(
            &output.handle,
            &output.strides,
            &output.shape.dims,
        )])
        .execute(WorkgroupLaunch::Output { pos: 0 });

    output
}

impl<R: Runtime, E: JitElement> GpuComputeShaderPhase for IntoContiguousEagerKernel<R, E> {
    fn compile(&self) -> ComputeShader {
        let mut scope = Scope::root();
        let item = E::gpu_elem().into();

        let tensor = Variable::GlobalInputArray(0, item);
        let output = Variable::GlobalOutputArray(0, item);

        IntoContiguousShader { tensor, output }.expand(&mut scope);

        scope.write_global_custom(output);

        let tensor = InputInfo::Array {
            item,
            visibility: Visibility::Read,
        };

        let out = OutputInfo::Array { item };

        let info = CompilationInfo {
            inputs: vec![tensor],
            outputs: vec![out],
            scope,
        };

        let settings = CompilationSettings::default();
        Compilation::new(info).compile(settings)
    }

    fn id(&self) -> String {
        format!("{:?}", core::any::TypeId::of::<Self>())
    }
}

impl IntoContiguousShader {
    pub(crate) fn expand(self, scope: &mut Scope) {
        let tensor = self.tensor;
        let id = Variable::Id;
        let output = self.output;

        let offset_input = scope.zero(Elem::UInt);

        IndexOffsetGlobalWithLayout {
            tensors: vec![tensor],
            indexes: vec![offset_input],
            layout: output,
            position: id,
            dim_start: 0u32.into(),
            dim_end: Variable::Rank,
        }
        .expand(scope);

        let value = scope.create_local(tensor.item());
        gpu!(scope, value = tensor[offset_input]);
        gpu!(scope, output[id] = value);
    }
}
