use crate::{
    element::JitElement, kernel::Kernel, ops::numeric::empty_device, tensor::JitTensor, JitRuntime,
};
use cubecl::frontend::{Numeric, Tensor, UInt, ABSOLUTE_POS};
use cubecl::ir::{
    Elem, IndexOffsetGlobalWithLayout, IntKind, Item, KernelDefinition, Scope, Variable, Visibility,
};
use cubecl::linalg::tensor::index_offset_with_layout;
use cubecl::prelude::*;
use cubecl::CubeDim;
use cubecl::{
    cpa, CubeCountSettings, Execution, InputInfo, KernelExpansion, KernelIntegrator,
    KernelSettings, OutputInfo,
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

#[cube(launch_unchecked)]
fn gather_kernel<T: Numeric, I: Int>(
    input: &Tensor<T>,
    input_layout: &Tensor<I>,
    indices: &Tensor<I>,
    output: &mut Tensor<T>,
    dim: &UInt,
) {
    let index = indices[ABSOLUTE_POS];

    let stride = input.stride(*dim);
    let mut offset = UInt::cast_from(index);
    offset *= stride;

    if *dim > 0 {
        let offset_before = index_offset_with_layout(
            input,
            input_layout,
            ABSOLUTE_POS,
            UInt::new(0),
            *dim,
            Comptime::new(false),
        );
        offset += offset_before;
    }

    let offset_after = index_offset_with_layout(
        input,
        input_layout,
        ABSOLUTE_POS,
        *dim + 1,
        input.rank(),
        Comptime::new(false),
    );
    offset += offset_after;
    output[ABSOLUTE_POS] = input[offset];
}

impl GatherComputeShader {
    pub fn expand(self, scope: &mut Scope) {
        match self.tensor {
            Variable::GlobalInputArray { .. } => (),
            Variable::GlobalOutputArray { .. } => (),
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

        let tensor = Variable::GlobalInputArray {
            id: 0,
            item: item_tensor,
        };
        let indices = scope.read_array(1, item_indices, Variable::AbsolutePos);

        let output_array = Variable::GlobalOutputArray {
            id: 0,
            item: item_tensor,
        };
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

    fn id(&self) -> cubecl::KernelId {
        cubecl::KernelId::new::<Self>().info(self.dim)
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

    Execution::start(kernel, tensor.client.clone())
        .inputs(&[tensor.as_handle_ref(), indices.as_handle_ref()])
        .outputs(&[output.as_handle_ref()])
        .execute(CubeCountSettings::Output { pos: 0 });

    output
}
