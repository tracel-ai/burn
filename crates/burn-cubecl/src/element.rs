use burn_backend::{Element, ElementComparison, bf16, f16};
use cubecl::{
    CubeElement as CubeElem, flex32,
    prelude::{Float, Int, Numeric},
};
use cubek::{
    matmul::definition::{MatmulPrecision, MatrixPrecision},
    reduce::ReducePrecision,
};

/// The base element trait for the jit backend.
pub trait CubeElement: Element + CubeElem + PartialEq + Numeric {}

/// Element that can be used for matrix multiplication. Includes ints and floats.
pub trait MatmulElement:
    CubeElement + MatmulPrecision<Acc: MatrixPrecision<Global: CubeElement>>
{
}

/// The float element type for the jit backend.
pub trait FloatElement: MatmulElement + Float + ElementComparison {}

/// The int element type for the jit backend.
pub trait IntElement:
    MatmulElement + Int + ReducePrecision<EI: CubeElement, EA: CubeElement> + ElementComparison
{
}

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
impl CubeElement for f16 {}
impl CubeElement for bf16 {}

impl FloatElement for f64 {}
impl FloatElement for f32 {}
impl FloatElement for flex32 {}
impl FloatElement for bf16 {}
impl FloatElement for f16 {}
impl IntElement for i64 {}
impl IntElement for i32 {}
impl IntElement for i16 {}
impl IntElement for i8 {}
impl IntElement for u64 {}
impl IntElement for u32 {}
impl IntElement for u16 {}
impl IntElement for u8 {}

impl BoolElement for u8 {}
impl BoolElement for u32 {}

impl MatmulElement for f64 {}
impl MatmulElement for f32 {}
impl MatmulElement for flex32 {}
impl MatmulElement for bf16 {}
impl MatmulElement for f16 {}

impl MatmulElement for i64 {}
impl MatmulElement for i32 {}
impl MatmulElement for i16 {}
impl MatmulElement for i8 {}
impl MatmulElement for u64 {}
impl MatmulElement for u32 {}
impl MatmulElement for u16 {}
impl MatmulElement for u8 {}
