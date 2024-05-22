use burn_cube::{CubeElement, Float, BF16, F16, F32};

/// The base element trait for the jit backend.
pub trait JitElement: burn_tensor::Element + CubeElement {}

/// The float element type for the jit backend.
pub trait FloatElement: JitElement {
    /// The associated Cube element for Cube kernels
    type CubeElement: Float;
}

/// The int element type for the jit backend.
pub trait IntElement: JitElement {}

impl JitElement for u32 {}

impl JitElement for i32 {}

impl JitElement for f32 {}

impl JitElement for half::f16 {}

impl JitElement for half::bf16 {}
impl FloatElement for f32 {
    type CubeElement = F32;
}
impl FloatElement for half::bf16 {
    type CubeElement = BF16;
}
impl FloatElement for half::f16 {
    type CubeElement = F16;
}
impl IntElement for i32 {}
