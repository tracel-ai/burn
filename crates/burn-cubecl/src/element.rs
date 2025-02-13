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

/// The element type for booleans for the jit backend.
pub trait BoolElement: JitElement + Int {
    /// The true value for the boolean element.
    fn true_val() -> Self {
        Self::from_int(1)
    }

    /// The false value for the boolean element.
    fn false_val() -> Self {
        Self::from_int(0)
    }

    /// New bool element from Rust bool.
    fn new_bool(val: bool) -> Self {
        match val {
            true => Self::true_val(),
            false => Self::false_val(),
        }
    }
}

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
impl IntElement for u32 {}

impl BoolElement for u8 {}
impl BoolElement for u32 {}
