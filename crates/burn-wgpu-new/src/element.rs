use burn_wgpu::JitElement;

use crate::compiler::wgsl;

/// The base element trait for the wgpu backend.
pub trait WgpuElement: JitElement {
    fn c_elem() -> wgsl::Elem;
}

/// The float element type for the wgpu backend.
pub trait FloatElement: WgpuElement + burn_wgpu::FloatElement {}

/// The int element type for the wgpu backend.
pub trait IntElement: WgpuElement + burn_wgpu::IntElement {}

impl WgpuElement for u32 {
    fn c_elem() -> wgsl::Elem {
        wgsl::Elem::U32
    }
}

impl WgpuElement for i32 {
    fn c_elem() -> wgsl::Elem {
        wgsl::Elem::U32
    }
}

impl WgpuElement for f32 {
    fn c_elem() -> wgsl::Elem {
        wgsl::Elem::F32
    }
}

impl FloatElement for f32 {}
impl IntElement for i32 {}
