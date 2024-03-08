use std::marker::PhantomData;

use crate::{
    codegen::{
        execute_dynamic, Compilation, CompilationInfo, CompilationSettings, EagerHandle, InputInfo,
        OutputInfo, WorkgroupLaunch,
    },
    gpu::{gpu, Elem, IndexOffsetGlobalWithLayout, Scope, Variable, Visibility},
    tensor::JitTensor,
    Compiler, JitElement, Runtime,
};

use super::{DynamicKernelSource, SourceTemplate};

pub(crate) struct IntoContiguousShader {
    tensor: Variable,
    output: Variable,
}

#[derive(new)]
pub(crate) struct IntoContiguousEagerKernel<R: Runtime, EO: JitElement> {
    _runtime: PhantomData<R>,
    _elem_out: PhantomData<EO>,
}

/// Make a jit tensor contiguous.
pub fn into_contiguous<R: Runtime, E: JitElement, const D: usize>(
    tensor: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    if tensor.is_contiguous() {
        return tensor;
    }

    let kernel = IntoContiguousEagerKernel::new();
    let num_elems = tensor.shape.num_elements();
    let buffer = tensor.client.empty(num_elems * core::mem::size_of::<E>());
    let output = JitTensor::new(
        tensor.client.clone(),
        tensor.device,
        tensor.shape.clone(),
        buffer,
    );

    execute_dynamic::<R, IntoContiguousEagerKernel<R, E>, u32>(
        &[EagerHandle::new(
            &tensor.handle,
            &tensor.strides,
            &tensor.shape.dims,
        )],
        &[EagerHandle::new(
            &output.handle,
            &output.strides,
            &output.shape.dims,
        )],
        None,
        kernel,
        WorkgroupLaunch::Output { pos: 0 },
        tensor.client,
    );

    output
}

impl<R: Runtime, E: JitElement> DynamicKernelSource for IntoContiguousEagerKernel<R, E> {
    fn source(&self) -> crate::kernel::SourceTemplate {
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
        let shader = Compilation::new(info).compile(settings);
        let shader = <R::Compiler as Compiler>::compile(shader);
        SourceTemplate::new(shader.to_string())
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

        // Batch offset for the lhs & rhs matrices.
        IndexOffsetGlobalWithLayout {
            tensors: vec![tensor],
            indexes: vec![offset_input],
            layout: output,
            index_ref: id,
            dim_start: 0u32.into(),
            dim_end: Variable::Rank,
        }
        .expand(scope);

        let value = scope.create_local(tensor.item());
        gpu!(scope, value = tensor[offset_input]);
        gpu!(scope, output[id] = value);
    }
}
