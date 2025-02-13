use cubecl::{
    flex32,
    prelude::{Float, Int, Numeric},
    CubeElement as CubeElem,
};

/// The base element trait for the jit backend.
pub trait CubeElement: burn_tensor::Element + CubeElem + PartialEq + Numeric {}

/// The float element type for the jit backend.
pub trait FloatElement: CubeElement + Float {}

/// The int element type for the jit backend.
pub trait IntElement: CubeElement + Int {}

/// The element type for booleans for the jit backend.
pub trait BoolElement: CubeElement + Int {
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

impl CubeElement for u64 {}
impl CubeElement for u32 {}
impl CubeElement for u16 {}
impl CubeElement for u8 {}
impl CubeElement for i64 {}
impl CubeElement for i32 {}
impl CubeElement for i16 {}
impl CubeElement for i8 {}
impl CubeElement for f64 {}
impl CubeElement for f32 {}
impl CubeElement for flex32 {}
impl CubeElement for half::f16 {}
impl CubeElement for half::bf16 {}

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
