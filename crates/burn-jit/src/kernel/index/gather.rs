use crate::codegen::dialect::gpu::{gpu, Elem, Scope, Variable};
use crate::codegen::Execution;
use crate::gpu::{ComputeShader, IntKind};
use crate::{
    codegen::{
        dialect::gpu, Compilation, CompilationInfo, CompilationSettings, EagerHandle, InputInfo,
        OutputInfo, WorkgroupLaunch,
    },
    element::JitElement,
    kernel::GpuComputeShaderPhase,
    ops::numeric::empty_device,
    tensor::JitTensor,
    Runtime,
};
use std::marker::PhantomData;

#[derive(new)]
struct GatherEagerKernel<R: Runtime, E: JitElement> {
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
        gpu!(scope, offset = cast(self.indices));
        gpu!(scope, stride = stride(tensor, self.dim));
        gpu!(scope, offset = offset * stride);

        // We fetch the offset before the `dim` dimension.
        if self.dim > 0 {
            let offset_before = scope.create_local(Elem::UInt);
            scope.index_offset_with_output_layout(gpu::IndexOffsetGlobalWithLayout {
                tensors: vec![tensor],
                indexes: vec![offset_before],
                layout: Variable::Id, // Will be updated.
                position: Variable::Id,
                dim_start: 0u32.into(),
                dim_end: self.dim.into(),
            });
            gpu!(scope, offset += offset_before);
        }

        let offset_after = scope.create_local(Elem::UInt);
        scope.index_offset_with_output_layout(gpu::IndexOffsetGlobalWithLayout {
            tensors: vec![tensor],
            indexes: vec![offset_after],
            layout: Variable::Id, // Will be updated.
            position: Variable::Id,
            dim_start: (self.dim + 1).into(),
            dim_end: Variable::Rank,
        });
        gpu!(scope, offset += offset_after);

        gpu!(scope, output = tensor[offset]);
    }
}

impl<R: Runtime, E: JitElement> GpuComputeShaderPhase for GatherEagerKernel<R, E> {
    fn compile(&self) -> ComputeShader {
        let mut scope = gpu::Scope::root();
        let item_tensor = E::gpu_elem().into();
        let item_indices: gpu::Item = gpu::Elem::Int(IntKind::I32).into();

        let tensor = gpu::Variable::GlobalInputArray(0, item_tensor);
        let indices = scope.read_array(1, item_indices, Variable::Id);

        let output_array = gpu::Variable::GlobalOutputArray(0, item_tensor);
        let output_local = scope.create_local(item_tensor);

        GatherComputeShader {
            tensor,
            indices,
            out: output_local,
            dim: self.dim,
        }
        .expand(&mut scope);

        scope.write_global(output_local, output_array, Variable::Id);

        let tensor = InputInfo::Array {
            item: item_tensor,
            visibility: gpu::Visibility::Read,
        };
        let indices = InputInfo::Array {
            item: gpu::Elem::Int(IntKind::I32).into(),
            visibility: gpu::Visibility::Read,
        };
        let out = OutputInfo::Array { item: item_tensor };

        let info = CompilationInfo {
            inputs: vec![tensor, indices],
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

pub(crate) fn gather<R: Runtime, E: JitElement, I: JitElement, const D: usize>(
    dim: usize,
    tensor: JitTensor<R, E, D>,
    indices: JitTensor<R, I, D>,
) -> JitTensor<R, E, D> {
    let shape_output = indices.shape.clone();
    let output = empty_device(tensor.client.clone(), tensor.device.clone(), shape_output);
    let kernel = GatherEagerKernel::<R, E>::new(dim);

    Execution::start(kernel, tensor.client)
        .inputs(&[
            EagerHandle::<R>::new(&tensor.handle, &tensor.strides, &tensor.shape.dims),
            EagerHandle::new(&indices.handle, &indices.strides, &indices.shape.dims),
        ])
        .outputs(&[EagerHandle::new(
            &output.handle,
            &output.strides,
            &output.shape.dims,
        )])
        .execute(WorkgroupLaunch::Output { pos: 0 });

    output
}
