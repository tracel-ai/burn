use crate::{compute::WorkGroup, gpu::ComputeShader};

#[cfg(target_family = "wasm")]
pub(crate) const WORKGROUP_DEFAULT: usize = 16;
#[cfg(not(target_family = "wasm"))]
pub(crate) const WORKGROUP_DEFAULT: usize = 32;

/// Dynamic jit kernel to create a [compute shader](ComputeShader).
pub trait GpuComputeShaderPhase: Send + Sync + 'static {
    /// Convert to compute shader
    fn compile(&self) -> ComputeShader;
    /// Identifier for the kernel, used for caching kernel compilation.
    fn id(&self) -> String {
        format!("{:?}", core::any::TypeId::of::<Self>())
    }
}

pub(crate) fn elemwise_workgroup(num_elems: usize, workgroup_size: usize) -> WorkGroup {
    let num_elem_per_invocation = workgroup_size * workgroup_size;
    let workgroups = f32::ceil(num_elems as f32 / num_elem_per_invocation as f32);
    let workgroup_x = f32::ceil(f32::sqrt(workgroups));
    let workgroup_y = f32::ceil(num_elems as f32 / (workgroup_x * num_elem_per_invocation as f32));

    WorkGroup::new(workgroup_x as u32, workgroup_y as u32, 1)
}
