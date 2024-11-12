use cubecl::{
    flex32,
    prelude::{Float, Int, Numeric},
    CubeElement,
};

/// The base element trait for the jit backend.
pub trait JitElement: burn_tensor::Element + CubeElement + PartialEq + Numeric {}

/// The float element type for the jit backend.
pub trait FloatElement: JitElement + Float {}

/// The int element type for the jit backend.
pub trait IntElement: JitElement + Int {}

impl JitElement for u64 {}
impl JitElement for u32 {}
impl JitElement for u16 {}
impl JitElement for u8 {}
impl JitElement for i64 {}
impl JitElement for i32 {}
impl JitElement for i16 {}
impl JitElement for i8 {}
impl JitElement for f64 {}
impl JitElement for f32 {}
impl JitElement for flex32 {}
impl JitElement for half::f16 {}
impl JitElement for half::bf16 {}

impl FloatElement for f64 {}
impl FloatElement for f32 {}
impl FloatElement for flex32 {}
impl FloatElement for half::bf16 {}
impl FloatElement for half::f16 {}
impl IntElement for i64 {}
impl IntElement for i32 {}
impl IntElement for i16 {}
impl IntElement for i8 {}
