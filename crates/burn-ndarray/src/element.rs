use burn_backend::{Element, ElementComparison};

use num_traits::Signed;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use num_traits::Pow;

use libm::{log1p, log1pf};

/// A float element for ndarray backend.
pub trait FloatNdArrayElement:
    NdArrayElement
    + Signed
    + core::cmp::PartialOrd<Self>
    + ExpElement
    + ElementComparison
    + bytemuck::Pod
where
    Self: Sized,
{
}

/// An int element for ndarray backend.
pub trait IntNdArrayElement: NdArrayElement + core::cmp::PartialOrd<Self> + ExpElement {}

/// A general element for ndarray backend.
pub trait NdArrayElement:
    Element
    + ndarray::LinalgScalar
    + ndarray::ScalarOperand
    + AddAssignElement
    + num_traits::FromPrimitive
    + core::ops::AddAssign
    + core::cmp::PartialEq
    + core::ops::Rem<Output = Self>
{
}

/// A element for ndarray backend that supports exp ops.
pub trait ExpElement {
    /// The output type of the `abs_elem` method. For most types, this will be the same as `Self`,
    /// but for some types (like complex numbers), it may an inner type.
    type AbsOutput: Element;
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
    fn abs_elem(self) -> Self::AbsOutput;
}

/// The addition assignment operator implemented for ndarray elements.
pub trait AddAssignElement<Rhs = Self> {
    /// Performs the addition assignment operation.
    ///
    /// For `bool`, this corresponds to logical OR assignment.
    fn add_assign(&mut self, rhs: Rhs);
}

impl<E: NdArrayElement> AddAssignElement for E {
    fn add_assign(&mut self, rhs: Self) {
        *self += rhs;
    }
}

impl AddAssignElement for bool {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self || rhs; // logical OR for bool
    }
}

/// A quantized element for the ndarray backend.
pub trait QuantElement: NdArrayElement {}

impl QuantElement for i8 {}

impl FloatNdArrayElement for f64 {}
impl FloatNdArrayElement for f32 {}

impl IntNdArrayElement for i64 {}
impl IntNdArrayElement for i32 {}
impl IntNdArrayElement for i16 {}
impl IntNdArrayElement for i8 {}

impl IntNdArrayElement for u64 {}
impl IntNdArrayElement for u32 {}
impl IntNdArrayElement for u16 {}
impl IntNdArrayElement for u8 {}

macro_rules! make_float {
    (
        $ty:ty,
        $log1p:expr
    ) => {
        impl NdArrayElement for $ty {}

        #[allow(clippy::cast_abs_to_unsigned)]
        impl ExpElement for $ty {
            type AbsOutput = Self;
            #[inline(always)]
            fn exp_elem(self) -> Self {
                self.exp()
            }

            #[inline(always)]
            fn log_elem(self) -> Self {
                self.ln()
            }

            #[inline(always)]
            fn log1p_elem(self) -> Self {
                $log1p(self)
            }

            #[inline(always)]
            fn powf_elem(self, value: f32) -> Self {
                self.pow(value)
            }

            #[inline(always)]
            fn powi_elem(self, value: i32) -> Self {
                #[cfg(feature = "std")]
                let val = self.powi(value);

                #[cfg(not(feature = "std"))]
                let val = Self::powf_elem(self, value as f32);

                val
            }

            #[inline(always)]
            fn sqrt_elem(self) -> Self {
                self.sqrt()
            }

            #[inline(always)]
            fn abs_elem(self) -> Self {
                self.abs()
            }
        }
    };
}
macro_rules! make_int {
    (
        $ty:ty,
        $abs:expr
    ) => {
        impl NdArrayElement for $ty {}

        #[allow(clippy::cast_abs_to_unsigned)]
        impl ExpElement for $ty {
            type AbsOutput = Self;
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
            fn abs_elem(self) -> Self::AbsOutput {
                $abs(self)
            }
        }
    };
}

make_float!(f64, log1p);
make_float!(f32, log1pf);

make_int!(i64, i64::wrapping_abs);
make_int!(i32, i32::wrapping_abs);
make_int!(i16, i16::wrapping_abs);
make_int!(i8, i8::wrapping_abs);
make_int!(u64, |x| x);
make_int!(u32, |x| x);
make_int!(u16, |x| x);
make_int!(u8, |x| x);

#[cfg(feature = "complex")]
mod complex {
    use super::*;
    use burn_complex::base::element::Complex;
    use num_traits::One;

    impl NdArrayElement for burn_complex::base::element::Complex<f32> {}
    impl NdArrayElement for burn_complex::base::element::Complex<f64> {}
    // where
    //     E: NdArrayElement + num_traits::Float + burn_backend::ElementOrdered + bytemuck::Pod
    // {
    // }

    impl<E: bytemuck::Pod + num_traits::Float + burn_backend::ElementOrdered> ExpElement
        for Complex<E>
    {
        type AbsOutput = E;

        fn exp_elem(self) -> Self {
            self.exp()
        }

        fn log_elem(self) -> Self {
            self.ln()
        }

        fn log1p_elem(self) -> Self {
            // Credit to soumyasen1809
            // https://github.com/rust-num/num-complex/pull/131
            (Self::one() + self).ln()
        }

        fn powf_elem(self, value: f32) -> Self {
            self.powf(E::from(value).expect("failed to convert to E"))
        }

        fn powi_elem(self, value: i32) -> Self {
            let mut output = self.powf(E::from(value).expect("failed to convert to E"));
            output.real = output.real.floor();
            output.imag = output.imag.floor();
            output
        }

        fn sqrt_elem(self) -> Self {
            self.sqrt()
        }

        fn abs_elem(self) -> Self::AbsOutput {
            self.abs()
        }
    }
}
