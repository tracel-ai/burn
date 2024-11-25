use cubecl::{
    flex32,
    prelude::{Algebraic, Float, Int, Numeric},
    CubeElement,
};

/// The base element trait for the jit backend.
pub trait BasicJitElement: burn_tensor::Element + CubeElement + Numeric {}

/// The base element trait for the jit backend.
pub trait JitElement: BasicJitElement + Algebraic {}

/// The float element type for the jit backend.
pub trait FloatElement: JitElement + Float {}

/// The int element type for the jit backend.
pub trait IntElement: JitElement + Int {}

impl BasicJitElement for u64 {}
impl BasicJitElement for u32 {}
impl BasicJitElement for u16 {}
impl BasicJitElement for u8 {}
impl BasicJitElement for i64 {}
impl BasicJitElement for i32 {}
impl BasicJitElement for i16 {}
impl BasicJitElement for i8 {}
impl BasicJitElement for f64 {}
impl BasicJitElement for f32 {}
impl BasicJitElement for flex32 {}
impl BasicJitElement for half::f16 {}
impl BasicJitElement for half::bf16 {}

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

impl BasicJitElement for bool {}

impl FloatElement for f64 {}
impl FloatElement for f32 {}
impl FloatElement for flex32 {}
impl FloatElement for half::bf16 {}
impl FloatElement for half::f16 {}
impl IntElement for i64 {}
impl IntElement for i32 {}
impl IntElement for i16 {}
impl IntElement for i8 {}
