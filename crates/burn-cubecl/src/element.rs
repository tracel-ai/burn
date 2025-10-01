use cubecl::{
    CubeElement as CubeElem, flex32,
    matmul::components::{MatmulPrecision, MatrixPrecision},
    prelude::{Float, Int, Numeric},
    reduce::ReducePrecision,
};

/// The base element trait for the jit backend.
pub trait CubeElement: burn_tensor::Element + CubeElem + PartialEq + Numeric {}

/// ELement that can be used for matrix multiplication. Includes ints and floats.
pub trait MatmulElement:
    CubeElement
    + MatmulPrecision<
        Lhs: MatrixPrecision,
        Rhs: MatrixPrecision,
        Acc: MatrixPrecision<Global: CubeElement>,
    >
{
}

/// The float element type for the jit backend.
pub trait FloatElement: MatmulElement + Float {}

/// The int element type for the jit backend.
pub trait IntElement:
    MatmulElement + Int + ReducePrecision<EI: CubeElement, EA: CubeElement>
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
impl IntElement for u64 {}
impl IntElement for u32 {}
impl IntElement for u16 {}
impl IntElement for u8 {}

impl BoolElement for u8 {}
impl BoolElement for u32 {}

impl MatmulElement for f64 {}
impl MatmulElement for f32 {}
impl MatmulElement for flex32 {}
impl MatmulElement for half::bf16 {}
impl MatmulElement for half::f16 {}

impl MatmulElement for i64 {}
impl MatmulElement for i32 {}
impl MatmulElement for i16 {}
impl MatmulElement for i8 {}
impl MatmulElement for u64 {}
impl MatmulElement for u32 {}
impl MatmulElement for u16 {}
impl MatmulElement for u8 {}
