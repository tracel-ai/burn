use burn_tensor::Element;
use ndarray::LinalgScalar;
use num_traits::{One, Signed, Zero};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use num_traits::Pow;

use libm::{log1p, log1pf};

/// A float element for ndarray backend.
pub trait FloatNdArrayElement: NdArrayElement + LinalgScalar + Signed
where
    Self: Sized,
{
}

/// An int element for ndarray backend.
pub trait IntNdArrayElement: NdArrayElement {}

/// A general element for ndarray backend.
pub trait NdArrayElement:
    Element
    + ndarray::LinalgScalar
    + ndarray::ScalarOperand
    + ExpElement
    + num_traits::FromPrimitive
    + core::ops::AddAssign
    + core::cmp::PartialEq
    + core::cmp::PartialOrd<Self>
    + core::ops::Rem<Output = Self>
{
}

/// A trait for computing the **sign** of a numeric value.
///
/// It returns:
///
/// - `1` if the number is positive  
/// - `-1` if the number is negative (only for signed types)  
/// - `0` if the number is zero  
///
/// For unsigned types, negative values are not possible, so the function only distinguishes
/// between `0` and nonzero values.
pub trait Signum: Zero + One {
    /// Returns the **sign** of the value as:
    /// - `1` for positive values,
    /// - `-1` for negative values (signed types only),
    /// - `0` for zero.
    fn signum(self) -> Self;
}

macro_rules! impl_signum_signed {
    ($($t:ty),*) => {
        $(
            impl Signum for $t {
                fn signum(self) -> Self {
                    if self > Self::zero() { Self::one() } else if self < Self::zero() { -Self::one() } else { Self::zero() }
                }
            }
        )*
    };
}

macro_rules! impl_signum_unsigned {
    ($($t:ty),*) => {
        $(
            impl Signum for $t {
                fn signum(self) -> Self {
                    if self > Self::zero() { Self::one() } else { Self::zero() }
                }
            }
        )*
    };
}

impl_signum_signed!(i32, i64, f32, f64); // for all required signed types
impl_signum_unsigned!(u8, u16, u32, u64); // // for all required unsigned types

/// A element for ndarray backend that supports exp ops.
pub trait ExpElement {
    /// Exponent
    fn exp_elem(self) -> Self;
    /// Log
    fn log_elem(self) -> Self;
    /// Log1p
    fn log1p_elem(self) -> Self;
    /// Powf
    fn powf_elem(self, value: f32) -> Self;
    /// Powi
    fn powi_elem(self, value: i32) -> Self;
    /// Sqrt
    fn sqrt_elem(self) -> Self;
    /// Abs
    fn abs_elem(self) -> Self;
    /// Abs for int
    fn int_abs_elem(self) -> Self;
}

/// A quantized element for the ndarray backend.
pub trait QuantElement: NdArrayElement {}

impl QuantElement for i8 {}

impl FloatNdArrayElement for f64 {}
impl FloatNdArrayElement for f32 {}

impl IntNdArrayElement for i64 {}
impl IntNdArrayElement for i32 {}
impl IntNdArrayElement for u8 {}

macro_rules! make_elem {
    (
        double
        $ty:ty
    ) => {
        impl NdArrayElement for $ty {}

        impl ExpElement for $ty {
            #[inline(always)]
            fn exp_elem(self) -> Self {
                (self as f64).exp() as $ty
            }

            #[inline(always)]
            fn log_elem(self) -> Self {
                (self as f64).ln() as $ty
            }

            #[inline(always)]
            fn log1p_elem(self) -> Self {
                log1p(self as f64) as $ty
            }

            #[inline(always)]
            fn powf_elem(self, value: f32) -> Self {
                (self as f64).pow(value) as $ty
            }

            #[inline(always)]
            fn powi_elem(self, value: i32) -> Self {
                #[cfg(feature = "std")]
                let val = f64::powi(self as f64, value) as $ty;

                #[cfg(not(feature = "std"))]
                let val = Self::powf_elem(self, value as f32);

                val
            }

            #[inline(always)]
            fn sqrt_elem(self) -> Self {
                (self as f64).sqrt() as $ty
            }

            #[inline(always)]
            fn abs_elem(self) -> Self {
                (self as f64).abs() as $ty
            }

            #[inline(always)]
            fn int_abs_elem(self) -> Self {
                (self as i64).abs() as $ty
            }
        }
    };
    (
        single
        $ty:ty
    ) => {
        impl NdArrayElement for $ty {}

        impl ExpElement for $ty {
            #[inline(always)]
            fn exp_elem(self) -> Self {
                (self as f32).exp() as $ty
            }

            #[inline(always)]
            fn log_elem(self) -> Self {
                (self as f32).ln() as $ty
            }

            #[inline(always)]
            fn log1p_elem(self) -> Self {
                log1pf(self as f32) as $ty
            }

            #[inline(always)]
            fn powf_elem(self, value: f32) -> Self {
                (self as f32).pow(value) as $ty
            }

            #[inline(always)]
            fn powi_elem(self, value: i32) -> Self {
                #[cfg(feature = "std")]
                let val = f32::powi(self as f32, value) as $ty;

                #[cfg(not(feature = "std"))]
                let val = Self::powf_elem(self, value as f32);

                val
            }

            #[inline(always)]
            fn sqrt_elem(self) -> Self {
                (self as f32).sqrt() as $ty
            }

            #[inline(always)]
            fn abs_elem(self) -> Self {
                (self as f32).abs() as $ty
            }

            #[inline(always)]
            fn int_abs_elem(self) -> Self {
                (self as i32).unsigned_abs() as $ty
            }
        }
    };
}

make_elem!(double f64);
make_elem!(double i64);
make_elem!(double u64);

make_elem!(single f32);
make_elem!(single i32);
make_elem!(single i16);
make_elem!(single i8);
make_elem!(single u32);
make_elem!(single u16);
make_elem!(single u8);
