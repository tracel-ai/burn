use burn_cube::{
    frontend::{CubeElem, Float, BF16, F16, F32, I32},
    CubeElement,
};

/// The base element trait for the jit backend.
pub trait JitElement: burn_tensor::Element + CubeElement {
    type CubeElement: CubeElem;
}

/// The float element type for the jit backend.
pub trait FloatElement: JitElement {
    type CubeElement: Float;
}

/// The int element type for the jit backend.
pub trait IntElement: JitElement {}

impl JitElement for u32 {
    type CubeElement = I32;
}

impl JitElement for i32 {
    type CubeElement = I32;
}

impl JitElement for f32 {
    type CubeElement = F32;
}

impl JitElement for half::f16 {
    type CubeElement = F16;
}

impl JitElement for half::bf16 {
    type CubeElement = BF16;
}
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
