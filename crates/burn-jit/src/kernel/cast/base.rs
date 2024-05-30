use burn_cube::{
    cpa,
    frontend::TensorHandle,
    ir::{KernelDefinition, Scope, Variable, Visibility},
    CubeCountSettings, Execution, InputInfo, KernelExpansion, KernelIntegrator, KernelSettings,
    OutputInfo,
};
use std::{any::TypeId, marker::PhantomData};

use crate::{kernel::Kernel, tensor::JitTensor, JitElement, JitRuntime};

/// Cast a tensor to the given element type.
///
/// Note: When input element is semantically a boolean, prefer bool_cast function.
pub fn cast<R: JitRuntime, EI: JitElement, EO: JitElement, const D: usize>(
    tensor: JitTensor<R, EI, D>,
) -> JitTensor<R, EO, D> {
    if TypeId::of::<EI>() == TypeId::of::<EO>() {
        return JitTensor::new(tensor.client, tensor.device, tensor.shape, tensor.handle);
    }

    let kernel = CastEagerKernel::<R, EI, EO>::new();
    let num_elems = tensor.shape.num_elements();
    let buffer = tensor.client.empty(num_elems * core::mem::size_of::<EO>());
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

pub(crate) struct CastShader {
    tensor: Variable,
    output: Variable,
}

#[derive(new)]
pub(crate) struct CastEagerKernel<R: JitRuntime, EI: JitElement, EO: JitElement> {
    _runtime: PhantomData<R>,
    _elem_in: PhantomData<EI>,
    _elem_out: PhantomData<EO>,
}

impl<R: JitRuntime, EI: JitElement, EO: JitElement> Kernel for CastEagerKernel<R, EI, EO> {
    fn define(&self) -> KernelDefinition {
        let mut scope = Scope::root();
        let item_input = EI::cube_elem().into();
        let item_output = EO::cube_elem().into();

        let tensor = Variable::GlobalInputArray(0, item_input);
        let output = Variable::GlobalOutputArray(0, item_output);

        CastShader { tensor, output }.expand(&mut scope);

        scope.write_global_custom(output);

        let tensor = InputInfo::Array {
            item: item_input,
            visibility: Visibility::Read,
        };

        let out = OutputInfo::Array { item: item_output };

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

impl CastShader {
    pub(crate) fn expand(self, scope: &mut Scope) {
        let tensor = self.tensor;
        let id = Variable::AbsolutePos;
        let output = self.output;

        let value = scope.create_local(output.item());
        cpa!(scope, value = tensor[id]);
        cpa!(scope, output[id] = value);
    }
}
