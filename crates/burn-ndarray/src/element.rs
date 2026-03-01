use burn_backend::Element;
#[cfg(feature = "complex")]
use burn_complex::base::element::{Complex32, Complex64};

use num_traits::Signed;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use num_traits::Pow;

use libm::{log1p, log1pf};

/// A float element for ndarray backend.
pub trait FloatNdArrayElement: NdArrayElement + Signed + core::cmp::PartialOrd<Self>
where
    Self: Sized,
{
}

/// An int element for ndarray backend.
pub trait IntNdArrayElement: NdArrayElement + core::cmp::PartialOrd<Self> {}

/// A general element for ndarray backend.
pub trait NdArrayElement:
    Element
    + ndarray::LinalgScalar
    + ndarray::ScalarOperand
    + ExpElement
    + AddAssignElement
    + num_traits::FromPrimitive
    + core::ops::AddAssign
    + core::cmp::PartialEq
    + core::ops::Rem<Output = Self>
{
}

/// A element for ndarray backend that supports exp ops.
pub trait ExpElement {
    type AbsOutput;
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

impl ExpElement for Complex32 {
    type AbsOutput = f32;

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
        self.powf(value)
    }

    // I have no idea if this is right or why one would even use powi with a complex number, I'll circle back
    // once everything else is working
    fn powi_elem(self, value: i32) -> Self {
        let mut output = self.powf(value as f32);
        output.real.floor();
        output.imag.floor();
        output
    }

    fn sqrt_elem(self) -> Self {
        self.sqrt()
    }

    fn abs_elem(self) -> Self::AbsOutput {
        self.abs()
    }
}

impl ExpElement for Complex64 {
    type AbsOutput = f64;

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
        self.powf(value.into())
    }

    fn powi_elem(self, value: i32) -> Self {
        let mut output = self.powf(value as f64);
        output.real.floor();
        output.imag.floor();
        output
    }

    fn sqrt_elem(self) -> Self {
        self.sqrt()
    }

    fn abs_elem(self) -> Self::AbsOutput {
        self.abs()
    }
}

///Mother fucker

/// A quantized element for the ndarray backend.
pub trait QuantElement: NdArrayElement {}

impl QuantElement for i8 {}
#[cfg(feature = "complex")]
impl NdArrayElement for burn_complex::base::element::Complex64 {}
#[cfg(feature = "complex")]
impl NdArrayElement for burn_complex::base::element::Complex32 {}

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
