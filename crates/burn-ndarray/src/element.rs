use burn_tensor::Element;
use ndarray::LinalgScalar;
use num_traits::{Signed, AsPrimitive};

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
impl FloatNdArrayElement for half::f16 {}
impl FloatNdArrayElement for half::bf16 {}

impl IntNdArrayElement for i64 {}
impl IntNdArrayElement for i32 {}
impl IntNdArrayElement for i16 {}
impl IntNdArrayElement for i8 {}

impl IntNdArrayElement for u64 {}
impl IntNdArrayElement for u32 {}
impl IntNdArrayElement for u16 {}
impl IntNdArrayElement for u8 {}

macro_rules! make_elem {
    (
        double
        $ty:ty
    ) => {
        impl NdArrayElement for $ty {}

        #[allow(clippy::cast_abs_to_unsigned)]
        impl ExpElement for $ty {
            #[inline(always)]
            fn exp_elem(self) -> Self {
                let self_f64: f64 = self.as_();
                self_f64.exp().as_()
            }

            #[inline(always)]
            fn log_elem(self) -> Self {
                let self_f64: f64 = self.as_();
                self_f64.ln().as_()
            }

            #[inline(always)]
            fn log1p_elem(self) -> Self {
                let self_f64: f64 = self.as_();
                log1p(self_f64).as_()
            }

            #[inline(always)]
            fn powf_elem(self, value: f32) -> Self {
                let self_f64: f64 = self.as_();
                self_f64.pow(value).as_()
            }

            #[inline(always)]
            fn powi_elem(self, value: i32) -> Self {
                #[cfg(feature = "std")]
                let val = {
                    let self_f64: f64 = self.as_();
                    let val = f64::powi(self_f64, value).as_();
                    val
                };

                #[cfg(not(feature = "std"))]
                let val = Self::powf_elem(self, value as f32);

                val
            }

            #[inline(always)]
            fn sqrt_elem(self) -> Self {
                let self_f64: f64 = self.as_();
                self_f64.sqrt().as_()
            }

            #[inline(always)]
            fn abs_elem(self) -> Self {
                let self_f64: f64 = self.as_();
                self_f64.abs().as_()
            }

            #[inline(always)]
            fn int_abs_elem(self) -> Self {
                let self_i64: i64 = self.as_();
                self_i64.abs().as_()
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
                let self_f32: f32 = self.as_();
                self_f32.exp().as_()
            }

            #[inline(always)]
            fn log_elem(self) -> Self {
                let self_f32: f32 = self.as_();
                self_f32.ln().as_()
            }

            #[inline(always)]
            fn log1p_elem(self) -> Self {
                let self_f32: f32 = self.as_();
                log1pf(self_f32).as_()
            }

            #[inline(always)]
            fn powf_elem(self, value: f32) -> Self {
                let self_f32: f32 = self.as_();
                self_f32.pow(value).as_()
            }

            #[inline(always)]
            fn powi_elem(self, value: i32) -> Self {
                #[cfg(feature = "std")]
                let val = {
                    let self_f32: f32 = self.as_();
                    let val = f32::powi(self_f32, value).as_();
                    val
                };

                #[cfg(not(feature = "std"))]
                let val = Self::powf_elem(self, value as f32);

                val
            }

            #[inline(always)]
            fn sqrt_elem(self) -> Self {
                let self_f32: f32 = self.as_();
                self_f32.sqrt().as_()
            }

            #[inline(always)]
            fn abs_elem(self) -> Self {
                let self_f32: f32 = self.as_();
                self_f32.abs().as_()
            }

            #[inline(always)]
            fn int_abs_elem(self) -> Self {
                let self_i32: i32 = self.as_();
                self_i32.unsigned_abs().as_()
            }
        }
    };
}

make_elem!(double f64);
make_elem!(double i64);
make_elem!(double u64);

make_elem!(single f32);
make_elem!(single i32);
make_elem!(single half::f16);
make_elem!(single half::bf16);
make_elem!(single i16);
make_elem!(single i8);
make_elem!(single u32);
make_elem!(single u16);
make_elem!(single u8);
