use std::{any::TypeId, marker::PhantomData};

use crate::{
    codegen::{
        execute_dynamic, Compilation, CompilationInfo, CompilationSettings, EagerHandle, InputInfo,
        OutputInfo, WorkgroupLaunch,
    },
    gpu::{gpu, Scope, Variable, Visibility},
    kernel::{DynamicKernelSource, SourceTemplate},
    tensor::JitTensor,
    Compiler, JitElement, Runtime,
};

/// Cast a tensor to the given element type.
///
/// Note: When input element is semantically a boolean, prefer bool_cast function.
pub fn cast<R: Runtime, EI: JitElement, EO: JitElement, const D: usize>(
    tensor: JitTensor<R, EI, D>,
) -> JitTensor<R, EO, D> {
    if TypeId::of::<EI>() == TypeId::of::<EO>() {
        return JitTensor::new(tensor.client, tensor.device, tensor.shape, tensor.handle);
    }

    let kernel = CastEagerKernel::new();
    let num_elems = tensor.shape.num_elements();
    let buffer = tensor.client.empty(num_elems * core::mem::size_of::<EO>());
    let output = JitTensor::new(
        tensor.client.clone(),
        tensor.device,
        tensor.shape.clone(),
        buffer,
    );

    execute_dynamic::<R, CastEagerKernel<R, EI, EO>, u32>(
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

pub(crate) struct CastShader {
    tensor: Variable,
    output: Variable,
}

#[derive(new)]
pub(crate) struct CastEagerKernel<R: Runtime, EI: JitElement, EO: JitElement> {
    _runtime: PhantomData<R>,
    _elem_in: PhantomData<EI>,
    _elem_out: PhantomData<EO>,
}

impl<R: Runtime, EI: JitElement, EO: JitElement> DynamicKernelSource
    for CastEagerKernel<R, EI, EO>
{
    fn source(&self) -> crate::kernel::SourceTemplate {
        let mut scope = Scope::root();
        let item_input = EI::gpu_elem().into();
        let item_output = EO::gpu_elem().into();

        let tensor = Variable::GlobalInputArray(0, item_input);
        let output = Variable::GlobalOutputArray(0, item_output);

        CastShader { tensor, output }.expand(&mut scope);

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
        format!("{:?}", core::any::TypeId::of::<Self>())
    }
}

impl CastShader {
    pub(crate) fn expand(self, scope: &mut Scope) {
        let tensor = self.tensor;
        let id = Variable::Id;
        let output = self.output;

        let value = scope.create_local(output.item());
        gpu!(scope, value = tensor[id]);
        gpu!(scope, output[id] = value);
    }
}
