use std::marker::PhantomData;

use burn_cube::{
    cpa,
    frontend::TensorHandle,
    ir::{Elem, IndexOffsetGlobalWithLayout, KernelDefinition, Scope, Variable, Visibility},
    CubeCountSettings, Execution, InputInfo, KernelExpansion, KernelIntegrator, KernelSettings,
    OutputInfo,
};

use crate::{tensor::JitTensor, JitElement, JitRuntime};

use super::Kernel;

pub(crate) struct IntoContiguousShader {
    tensor: Variable,
    output: Variable,
}

#[derive(new)]
pub(crate) struct IntoContiguousEagerKernel<R: JitRuntime, E: JitElement> {
    _runtime: PhantomData<R>,
    _elem_out: PhantomData<E>,
}

/// Make a jit tensor contiguous.
pub fn into_contiguous<R: JitRuntime, E: JitElement, const D: usize>(
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
        .inputs(&[TensorHandle::<R>::new(
            &tensor.handle,
            &tensor.strides,
            &tensor.shape.dims,
        )])
        .outputs(&[TensorHandle::new(
            &output.handle,
            &output.strides,
            &output.shape.dims,
        )])
        .execute(CubeCountSettings::Output { pos: 0 });

    output
}

impl<R: JitRuntime, E: JitElement> Kernel for IntoContiguousEagerKernel<R, E> {
    fn define(&self) -> KernelDefinition {
        let mut scope = Scope::root();
        let item = E::cube_elem().into();

        let tensor = Variable::GlobalInputArray(0, item);
        let output = Variable::GlobalOutputArray(0, item);

        IntoContiguousShader { tensor, output }.expand(&mut scope);

        scope.write_global_custom(output);

        let tensor = InputInfo::Array {
            item,
            visibility: Visibility::Read,
        };

        let out = OutputInfo::Array { item };

        let info = KernelExpansion {
            inputs: vec![tensor],
            outputs: vec![out],
            scope,
        };

        let settings = KernelSettings::default();
        KernelIntegrator::new(info).integrate(settings)
    }

    fn id(&self) -> String {
        format!("{:?}", core::any::TypeId::of::<Self>())
    }
}

impl IntoContiguousShader {
    pub(crate) fn expand(self, scope: &mut Scope) {
        let tensor = self.tensor;
        let id = Variable::AbsolutePos;
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
        cpa!(scope, value = tensor[offset_input]);
        cpa!(scope, output[id] = value);
    }
}
