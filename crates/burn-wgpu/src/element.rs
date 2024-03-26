use burn_jit::JitElement;

use crate::compiler::wgsl;

/// The base element trait for the wgpu backend.
pub trait WgpuElement: JitElement {
    fn wgpu_elem() -> wgsl::Elem;
}

/// The float element type for the wgpu backend.
pub trait FloatElement: WgpuElement + burn_jit::FloatElement {}

/// The int element type for the wgpu backend.
pub trait IntElement: WgpuElement + burn_jit::IntElement {}

impl WgpuElement for u32 {
    fn wgpu_elem() -> wgsl::Elem {
        wgsl::Elem::U32
    }
}

impl WgpuElement for i32 {
    fn wgpu_elem() -> wgsl::Elem {
        wgsl::Elem::I32
    }
}

impl WgpuElement for f32 {
    fn wgpu_elem() -> wgsl::Elem {
        wgsl::Elem::F32
    }
}

impl FloatElement for f32 {}
impl IntElement for i32 {}
