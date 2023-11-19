use burn_tensor::Element;
use libm::{exp, fabs, log, log1p, pow, sqrt};
use libm::{expf, fabsf, log1pf, logf, powf, sqrtf};
use ndarray::LinalgScalar;

/// A float element for ndarray backend.
pub trait FloatNdArrayElement: NdArrayElement + LinalgScalar
where
    Self: Sized,
{
}

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
{
}

/// A element for ndarray backend that supports exp ops.
pub trait ExpElement {
    fn exp_elem(self) -> Self;
    fn log_elem(self) -> Self;
    fn log1p_elem(self) -> Self;
    fn powf_elem(self, value: f32) -> Self;
    fn powi_elem(self, value: i32) -> Self;
    fn sqrt_elem(self) -> Self;
    fn abs_elem(self) -> Self;
    fn int_abs_elem(self) -> Self;
}

impl FloatNdArrayElement for f64 {}
impl FloatNdArrayElement for f32 {}

macro_rules! make_elem {
    (
        double
        $ty:ty
    ) => {
        impl NdArrayElement for $ty {}

        impl ExpElement for $ty {
            #[inline(always)]
            fn exp_elem(self) -> Self {
                exp(self as f64) as $ty
            }

            #[inline(always)]
            fn log_elem(self) -> Self {
                log(self as f64) as $ty
            }

            #[inline(always)]
            fn log1p_elem(self) -> Self {
                log1p(self as f64) as $ty
            }

            #[inline(always)]
            fn powf_elem(self, value: f32) -> Self {
                pow(self as f64, value.into()) as $ty
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
                sqrt(self as f64) as $ty
            }

            #[inline(always)]
            fn abs_elem(self) -> Self {
                fabs(self as f64) as $ty
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
                expf(self as f32) as $ty
            }

            #[inline(always)]
            fn log_elem(self) -> Self {
                logf(self as f32) as $ty
            }

            #[inline(always)]
            fn log1p_elem(self) -> Self {
                log1pf(self as f32) as $ty
            }

            #[inline(always)]
            fn powf_elem(self, value: f32) -> Self {
                powf(self as f32, value.into()) as $ty
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
                sqrtf(self as f32) as $ty
            }

            #[inline(always)]
            fn abs_elem(self) -> Self {
                fabsf(self as f32) as $ty
            }

            #[inline(always)]
            fn int_abs_elem(self) -> Self {
                (self as i32).abs() as $ty
            }
        }
    };
}

make_elem!(double f64);
make_elem!(double i64);

make_elem!(single f32);
make_elem!(single i32);
make_elem!(single i16);
make_elem!(single u8);
