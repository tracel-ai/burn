use burn_jit::JitElement;

use crate::compiler;

/// The base element trait for the wgpu backend.
pub trait CudaElement: JitElement {
    fn cuda_elem() -> compiler::Elem;
}

/// The float element type for the wgpu backend.
pub trait FloatElement: CudaElement + burn_jit::FloatElement {}

/// The int element type for the wgpu backend.
pub trait IntElement: CudaElement + burn_jit::IntElement {}

impl CudaElement for u32 {
    fn cuda_elem() -> compiler::Elem {
        compiler::Elem::U32
    }
}

impl CudaElement for i32 {
    fn cuda_elem() -> compiler::Elem {
        compiler::Elem::I32
    }
}

impl CudaElement for f32 {
    fn cuda_elem() -> compiler::Elem {
        compiler::Elem::F32
    }
}

impl CudaElement for half::bf16 {
    fn cuda_elem() -> compiler::Elem {
        compiler::Elem::BF16
    }
}

impl FloatElement for f32 {}
impl FloatElement for half::bf16 {}
impl IntElement for i32 {}
