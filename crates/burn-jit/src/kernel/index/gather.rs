use crate::{
    element::JitElement, kernel::Kernel, ops::numeric::empty_device, tensor::JitTensor, JitRuntime,
};
use burn_cube::ir::{
    Elem, IndexOffsetGlobalWithLayout, IntKind, Item, KernelDefinition, Scope, Variable, Visibility,
};
use burn_cube::{
    cpa, frontend::TensorHandle, CubeCountSettings, Execution, InputInfo, KernelExpansion,
    KernelIntegrator, KernelSettings, OutputInfo,
};
use std::marker::PhantomData;

#[derive(new)]
struct GatherEagerKernel<R: JitRuntime, E: JitElement> {
    dim: usize,
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

struct GatherComputeShader {
    tensor: Variable,
    indices: Variable,
    out: Variable,
    dim: usize,
}

impl GatherComputeShader {
    pub fn expand(self, scope: &mut Scope) {
        match self.tensor {
            Variable::GlobalInputArray(_, _) => (),
            Variable::GlobalOutputArray(_, _) => (),
            _ => panic!("Tensor variable must be an global array."),
        };

        let tensor = self.tensor;
        let output = self.out;

        let stride = scope.create_local(Elem::UInt);
        let offset = scope.create_local(Elem::UInt);

        // The offset of the `dim` dimension is obtained by the indices tensor.
        cpa!(scope, offset = cast(self.indices));
        cpa!(scope, stride = stride(tensor, self.dim));
        cpa!(scope, offset = offset * stride);

        // We fetch the offset before the `dim` dimension.
        if self.dim > 0 {
            let offset_before = scope.create_local(Elem::UInt);
            scope.index_offset_with_output_layout(IndexOffsetGlobalWithLayout {
                tensors: vec![tensor],
                indexes: vec![offset_before],
                layout: Variable::AbsolutePos, // Will be updated.
                position: Variable::AbsolutePos,
                dim_start: 0u32.into(),
                dim_end: self.dim.into(),
            });
            cpa!(scope, offset += offset_before);
        }

        let offset_after = scope.create_local(Elem::UInt);
        scope.index_offset_with_output_layout(IndexOffsetGlobalWithLayout {
            tensors: vec![tensor],
            indexes: vec![offset_after],
            layout: Variable::AbsolutePos, // Will be updated.
            position: Variable::AbsolutePos,
            dim_start: (self.dim + 1).into(),
            dim_end: Variable::Rank,
        });
        cpa!(scope, offset += offset_after);

        cpa!(scope, output = tensor[offset]);
    }
}

impl<R: JitRuntime, E: JitElement> Kernel for GatherEagerKernel<R, E> {
    fn define(&self) -> KernelDefinition {
        let mut scope = Scope::root();
        let item_tensor = E::cube_elem().into();
        let item_indices: Item = Elem::Int(IntKind::I32).into();

        let tensor = Variable::GlobalInputArray(0, item_tensor);
        let indices = scope.read_array(1, item_indices, Variable::AbsolutePos);

        let output_array = Variable::GlobalOutputArray(0, item_tensor);
        let output_local = scope.create_local(item_tensor);

        GatherComputeShader {
            tensor,
            indices,
            out: output_local,
            dim: self.dim,
        }
        .expand(&mut scope);

        scope.write_global(output_local, output_array, Variable::AbsolutePos);

        let tensor = InputInfo::Array {
            item: item_tensor,
            visibility: Visibility::Read,
        };
        let indices = InputInfo::Array {
            item: Elem::Int(IntKind::I32).into(),
            visibility: Visibility::Read,
        };
        let out = OutputInfo::Array { item: item_tensor };

        let info = KernelExpansion {
            inputs: vec![tensor, indices],
            outputs: vec![out],
            scope,
        };

        let settings = KernelSettings::default();
        KernelIntegrator::new(info).integrate(settings)
    }

    fn id(&self) -> String {
        format!("{:?}dim={}", core::any::TypeId::of::<Self>(), self.dim)
    }
}

pub(crate) fn gather<R: JitRuntime, E: JitElement, I: JitElement, const D: usize>(
    dim: usize,
    tensor: JitTensor<R, E, D>,
    indices: JitTensor<R, I, D>,
) -> JitTensor<R, E, D> {
    let shape_output = indices.shape.clone();
    let output = empty_device(tensor.client.clone(), tensor.device.clone(), shape_output);
    let kernel = GatherEagerKernel::<R, E>::new(dim);

    Execution::start(kernel, tensor.client)
        .inputs(&[
            TensorHandle::<R>::new(&tensor.handle, &tensor.strides, &tensor.shape.dims),
            TensorHandle::new(&indices.handle, &indices.strides, &indices.shape.dims),
        ])
        .outputs(&[TensorHandle::new(
            &output.handle,
            &output.strides,
            &output.shape.dims,
        )])
        .execute(CubeCountSettings::Output { pos: 0 });

    output
}
