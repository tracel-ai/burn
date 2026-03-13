//use num_complex::Complex as NumComplex;

/// 32-bit complex number type (real and imaginary parts are f32).
use burn_tensor::{
    DType, Distribution, Element, ElementConversion, ElementEq, ElementLimits, ElementRandom,
    cast::ToElement,
};

use core::ops::{AddAssign, Rem};
use num_traits::FromPrimitive;
use num_traits::Num;
use num_traits::One;
use num_traits::Pow;
use num_traits::Zero;
use num_traits::float::FloatCore;
use num_traits::identities::ConstZero;
use rand::Rng;
#[cfg(feature = "ndarray")]
mod ndarray {
    use super::{Complex32, Complex64};
    use ndarray::ScalarOperand;
    impl ScalarOperand for Complex32 {}
    impl ScalarOperand for Complex64 {}
}

#[cfg(feature = "tch")]
mod tch {
    use super::{Complex32, Complex64};
    use tch::kind::Element as TchElement;
    impl TchElement for Complex32 {}
    impl TchElement for Complex64 {}
}

use std::ops::Div;
pub trait ToComplex<C> {
    fn to_complex(&self) -> C;
}
use paste::paste;

use crate::base::ComplexTensorOps;
pub trait ToComplexElement: ToElement + ToComplex<Complex32> + ToComplex<Complex64> {
    fn to_complex32(&self) -> Complex32 {
        self.to_complex()
    }
    fn to_complex64(&self) -> Complex64 {
        self.to_complex()
    }
}

// will attempt after I get ndarray to compile

// pub struct Complex<C> {
//     pub real: C,
//     pub imag: C,
// }

// impl<C> Complex<C> {
//     #[inline]
//     pub fn new(real: C, imag: C) -> Self {
//         Self { real, imag }
//     }
// }

/// Macro to implement the element trait for a type.
#[macro_export]
macro_rules! make_complex {
    (
        ty $type:ident $inner:ident $precision:expr,
        random $random:expr,
        cmp $cmp:expr,
        dtype $dtype:expr
    ) => {
        make_complex!(ty $type $inner $precision, convert $convert, random $random, cmp $cmp, dtype $dtype, min $type::MIN, max $type::MAX);
    };
    (
        ty $type:ident $inner:ident $precision:expr,
        convert $convert:expr,
        random $random:expr,
        cmp $cmp:expr,
        dtype $dtype:expr,
        min $min:expr,
        max $max:expr
    ) => {
        #[derive(Debug, Clone, Copy, PartialEq, Default, bytemuck::Pod, bytemuck::Zeroable)]
        #[repr(C)]
        pub struct $type {
            /// Real component
            pub real: $inner,
            /// Imaginary component
            pub imag: $inner,
        }

        impl $type {
            /// Create a new complex number from real and imaginary parts
            #[inline]
            pub const fn new(real: $inner, imag: $inner) -> Self {
                Self { real, imag }
            }

            /// Create a complex number from a real number
            #[inline]
            pub const fn from_real(real: $inner) -> Self {
                Self { real, imag: $inner::ZERO }
            }

            /// Create a complex number from any element primitive
            #[inline]
            pub fn from_elem<E: ToElement>(real: E) -> Self {
                paste! {
                    Self { real: real.[<to_ $inner>](), imag: $inner::ZERO }
                }
            }

            /// Get the magnitude (absolute value) of the complex number
            #[inline]
            pub fn abs(self) -> $inner {
                (self.real * self.real + self.imag * self.imag).sqrt()
            }

            /// Get the conjugate of the complex number
            #[inline]
            pub fn conj(self) -> Self {
                Self {
                    real: self.real,
                    imag: -self.imag,
                }
            }

            #[inline]
            pub fn dtype() -> DType {
                DType::$type
            }

            // The below methods are copied from num_complex 0.4.6, since we can't implement the required element traits for num_complex::Complex.
            // link to the docs: https://docs.rs/num-complex/0.4.6/num_complex/
            // Credit to https://github.com/cuviper for the original implementations.

            /// Computes `e^(self)`, where `e` is the base of the natural logarithm.
            #[inline]
            pub fn exp(self) -> Self {
                // formula: e^(a + bi) = e^a (cos(b) + i*sin(b)) = from_polar(e^a, b)

                let $type { real, mut imag } = self;
                // Treat the corner cases +∞, -∞, and NaN
                if real.is_infinite() {
                    if real < $inner::zero() {
                        if !imag.is_finite() {
                            return Self::new($inner::zero(), $inner::zero());
                        }
                    } else if imag == $inner::zero() || !imag.is_finite() {
                        if imag.is_infinite() {
                            imag = $inner::nan();
                        }
                        return Self::new(real, imag);
                    }
                } else if real.is_nan() && imag == $inner::zero() {
                    return self;
                }

                Self::from_polar(real.exp(), imag)
            }
            /// Convert a polar representation into a complex number.
            #[inline]
            pub fn from_polar(r: $inner, theta: $inner) -> Self {
                Self::new(r * theta.cos(), r * theta.sin())

            }

            /// Calculate |self|
            #[inline]
            pub fn norm(self) -> $inner {
                self.real.hypot(self.imag)
            }

            /// Convert to polar form (r, theta), such that
            /// `self = r * exp(i * theta)`
            #[inline]
            pub fn to_polar(self) -> ($inner, $inner) {
                (self.norm(), self.arg())
            }

            /// Calculate the principal Arg of self.
            #[inline]
            pub fn arg(self) -> $inner {
                self.imag.atan2(self.real)
            }

            /// Returns the logarithm of `self` with respect to an arbitrary base.
            #[inline]
            pub fn log(self, base: $inner) -> Self {
                // formula: log_y(x) = log_y(ρ e^(i θ))
                // = log_y(ρ) + log_y(e^(i θ)) = log_y(ρ) + ln(e^(i θ)) / ln(y)
                // = log_y(ρ) + i θ / ln(y)
                let (r, theta) = self.to_polar();
                Self::new(r.log(base), theta / base.ln())
            }

            /// Computes the principal value of the inverse tangent of `self`.
            ///
            /// This function has two branch cuts:
            ///
            /// * `(-∞i, -i]`, continuous from the left.
            /// * `[i, ∞i)`, continuous from the right.
            ///
            /// The branch satisfies `-π/2 ≤ Re(atan(z)) ≤ π/2`.
            #[inline]
            pub fn atan(self) -> Self {
                // formula: arctan(z) = (ln(1+iz) - ln(1-iz))/(2i)
                let i = Self::i();
                let one = Self::one();
                let two = one + one;
                if self == i {
                    return Self::new($inner::zero(), $inner::infinity());
                } else if self == -i {
                    return Self::new($inner::zero(), -$inner::infinity());
                }
                ((one + i * self).ln() - (one - i * self).ln()) / (two * i)
            }

            /// Computes the principal value of natural logarithm of `self`.
            ///
            /// This function has one branch cut:
            ///
            /// * `(-∞, 0]`, continuous from above.
            ///
            /// The branch satisfies `-π ≤ arg(ln(z)) ≤ π`.
            #[inline]
            pub fn ln(self) -> Self {
                // formula: ln(z) = ln|z| + i*arg(z)
                let (r, theta) = self.to_polar();
                Self::new(r.ln(), theta)
            }

            #[inline]
            pub fn one() -> Self {
                Self::new($inner::one(), $inner::zero())
            }

            /// Returns the imaginary unit.
            ///
            /// See also [`Complex::I`].
            #[inline]
            pub fn i() -> Self {
                Self::new($inner::zero(), $inner::one())
            }

            /// Returns the square of the norm (since `T` doesn't necessarily
            /// have a sqrt function), i.e. `re^2 + im^2`.
            #[inline]
            pub fn norm_sqr(&self) -> $inner {
                self.real.clone() * self.real.clone() + self.imag.clone() * self.imag.clone()
            }

            /// Raises `self` to a floating point power.
            #[inline]
            pub fn powf(self, exp: $inner) -> Self {
            if exp.is_zero() {
                return Self::one();
            }
            // formula: x^y = (ρ e^(i θ))^y = ρ^y e^(i θ y)
            // = from_polar(ρ^y, θ y)
            let (r, theta) = self.to_polar();
            Self::from_polar(r.powf(exp), theta * exp)
            }

            /// Raises `self` to a complex power.
            #[inline]
            pub fn powc(self, exp: Self) -> Self {
                if exp == $type::new($inner::zero(), $inner::zero()) {
                    return Self::one();
                }
                // formula: x^y = exp(y * ln(x))
                (exp * self.ln()).exp()
            }

            /// Computes the principal value of the square root of `self`.
            ///
            /// This function has one branch cut:
            ///
            /// * `(-∞, 0)`, continuous from above.
            ///
            /// The branch satisfies `-π/2 ≤ arg(sqrt(z)) ≤ π/2`.
            #[inline]
            pub fn sqrt(self) -> Self {
                if self.imag.is_zero() {
                    if self.real.is_sign_positive() {
                        // simple positive real √r, and copy `im` for its sign
                        Self::new(self.real.sqrt(), self.imag)
                    } else {
                        // √(r e^(iπ)) = √r e^(iπ/2) = i√r
                        // √(r e^(-iπ)) = √r e^(-iπ/2) = -i√r
                        let re = $inner::zero();
                        let im = (-self.real).sqrt();
                        if self.imag.is_sign_positive() {
                            Self::new(re, im)
                        } else {
                            Self::new(re, -im)
                        }
                    }
                } else if self.real.is_zero() {
                    // √(r e^(iπ/2)) = √r e^(iπ/4) = √(r/2) + i√(r/2)
                    // √(r e^(-iπ/2)) = √r e^(-iπ/4) = √(r/2) - i√(r/2)
                    let one = $inner::one();
                    let two = one + one;
                    let x = (self.imag.abs() / two).sqrt();
                    if self.imag.is_sign_positive() {
                        Self::new(x, x)
                    } else {
                        Self::new(x, -x)
                    }
                } else {
                    // formula: sqrt(r e^(it)) = sqrt(r) e^(it/2)
                    let one = $inner::one();
                    let two = one + one;
                    let (r, theta) = self.to_polar();
                    Self::from_polar(r.sqrt(), theta / two)
                }
            }


        }


        impl One for $type {
            fn one() -> Self {
                Self::one()
            }
        }

        impl Zero for $type {
            fn zero() -> Self {
                Self::new($inner::zero(), $inner::zero())
            }
            fn is_zero(&self) -> bool {
                self.real.is_zero() && self.imag.is_zero()
            }
        }



        // impl<$type> ToComplex<Complex32> for $type {
        //     fn to_complex(&self) -> Complex32 {
        //         Complex32::new(self.real as f32, self.imag as f32)
        //     }
        // }

        // impl ToComplex<Complex64> for $type {
        //     fn to_complex(&self) -> Complex64 {
        //         Complex64::new(self.real as f64, self.imag as f64)
        //     }
        // }

        impl core::fmt::Display for $type {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                if self.imag >= 0.0 {
                    write!(f, "{}+{}i", self.real, self.imag)
                } else {
                    write!(f, "{}{}i", self.real, self.imag)
                }
            }
        }


        // Arithmetic operators for Complex32
        impl core::ops::Add for $type {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                Self {
                    real: self.real + rhs.real,
                    imag: self.imag + rhs.imag,
                }
            }
        }

        impl core::ops::Sub for $type {
            type Output = Self;

            fn sub(self, rhs: Self) -> Self::Output {
                Self {
                    real: self.real - rhs.real,
                    imag: self.imag - rhs.imag,
                }
            }
        }

        impl core::ops::Mul for $type {
            type Output = Self;

            fn mul(self, rhs: Self) -> Self::Output {
                Self {
                    real: self.real * rhs.real - self.imag * rhs.imag,
                    imag: self.real * rhs.imag + self.imag * rhs.real,
                }
            }
        }

        impl core::ops::Neg for $type {
            type Output = Self;

            fn neg(self) -> Self::Output {
                Self {
                    real: -self.real,
                    imag: -self.imag,
                }
            }
        }


        impl ElementConversion for $type {
            #[inline(always)]
            fn from_elem<E: ToElement>(elem: E) -> Self {
                Self::from_elem(elem)
            }
            #[inline(always)]
            fn elem<E: Element>(self) -> E {
                E::from_elem(self)
            }
        }



        impl ElementRandom for $type {
            fn random<R: Rng>(distribution: Distribution, rng: &mut R) -> Self {
                #[allow(clippy::redundant_closure_call)]
                $random(distribution, rng)
            }
        }

        impl ElementLimits for $type {
            const MIN: Self = $min;
            const MAX: Self = $max;
        }

        impl ToElement for $type {
            #[inline]
            fn to_i64(&self) -> i64 {
                self.real.to_i64()
            }
            #[inline]
            fn to_u64(&self) -> u64 {
                self.real.to_u64()
            }
            #[inline]
            fn to_f32(&self) -> f32 {
                self.real as f32
            }
            #[inline]
            fn to_f64(&self) -> f64 {
                self.real as f64
            }
            #[inline]
            fn to_bool(&self) -> bool {
                self.real != 0.0 || self.imag != 0.0
            }
        }
        impl ToComplexElement for $type {
            #[inline]
            fn to_complex32(&self) -> Complex32 {
                Complex32::new(self.real as f32, self.imag as f32)
            }
            #[inline]
            fn to_complex64(&self) -> Complex64 {
                Complex64::new(self.real as f64, self.imag as f64)
            }
        }

        impl ElementEq for $type {
            fn eq(&self, other: &Self) -> bool {
                self.real == other.real && self.imag == other.imag
            }
        }

        impl Element for $type {
            fn dtype() -> DType {
                $dtype
            }
        }

        impl AddAssign for $type {
            fn add_assign(&mut self, rhs: Self) {
                self.real += rhs.real;
                self.imag += rhs.imag;
            }
        }

        impl Rem for $type {
            type Output = Self;

            fn rem(self, rhs: Self) -> Self::Output {
                Self {
                    real: self.real % rhs.real,
                    imag: self.imag % rhs.imag,
                }
            }
        }

        impl FromPrimitive for $type {
            fn from_i64(n: i64) -> Option<Self> {
                Some(Self::from_real(n as $inner))
            }
            fn from_u64(n: u64) -> Option<Self> {
                Some(Self::from_real(n as $inner))
            }
            fn from_f32(n: f32) -> Option<Self> {
                Some(Self::from_real(n as $inner))
            }
            fn from_f64(n: f64) -> Option<Self> {
                Some(Self::from_real(n as $inner))
            }
        }
    };
}

make_complex!(
    ty Complex32 f32 Precision::Full,
    convert ToComplexElement::to_complex32,
    random |distribution: Distribution, rng: &mut R| {
        let real: f32 = distribution.sampler(rng).sample();
        let imag: f32 = distribution.sampler(rng).sample();
        Complex32::new(real, imag)
    },
    cmp |a: &Complex32, b: &Complex32| {
        // Compare by magnitude, then by real part if magnitudes are equal
        let mag_cmp = a.abs().total_cmp(&b.abs());
        if mag_cmp == Ordering::Equal {
            a.real.total_cmp(&b.real)
        } else {
            mag_cmp
        }
    },
    dtype DType::Complex32,
    min Complex32::new(f32::MIN, f32::MIN),
    max Complex32::new(f32::MAX, f32::MAX)
);
macro_rules! to_complex {
    (
        $type:ident
    ) => {
        impl ToComplex<Complex32> for $type {
            #[inline]
            fn to_complex(&self) -> Complex32 {
                Complex32::new(*self as f32, 0.0)
            }
        }
        impl ToComplex<Complex64> for $type {
            #[inline]
            fn to_complex(&self) -> Complex64 {
                Complex64::new(*self as f64, 0.0)
            }
        }
    };
}

to_complex!(i64);
to_complex!(i32);
to_complex!(f32);
to_complex!(f64);
//to_complex!(bool);

make_complex!(
    ty Complex64 f64 Precision::Double,
    convert ToComplexElement::to_complex64,
    random |distribution: Distribution, rng: &mut R| {
        let real: f64 = distribution.sampler(rng).sample();
        let imag: f64 = distribution.sampler(rng).sample();
        Complex64::new(real, imag)
    },
    cmp |a: &Complex64, b: &Complex64| {
        // Compare by magnitude, then by real part if magnitudes are equal
        let mag_cmp = a.abs().total_cmp(&b.abs());
        if mag_cmp == Ordering::Equal {
            a.real.total_cmp(&b.real)
        } else {
            mag_cmp
        }
    },
    dtype DType::Complex64,
    min Complex64::new(f64::MIN, f64::MIN),
    max Complex64::new(f64::MAX, f64::MAX)


);

// (a + i b) / (c + i d) == [(a + i b) * (c - i d)] / (c*c + d*d)
//   == [(a*c + b*d) / (c*c + d*d)] + i [(b*c - a*d) / (c*c + d*d)]
impl Div<Complex32> for Complex32 {
    type Output = Self;

    #[inline]
    fn div(self, other: Self) -> Self::Output {
        let norm_sqr = other.norm_sqr();
        let re = self.real.clone() * other.real + self.imag.clone() * other.imag;
        let im = self.imag * other.real - self.real * other.imag;
        Self::Output::new(re / norm_sqr.clone(), im / norm_sqr)
    }
}

// (a + i b) / (c + i d) == [(a + i b) * (c - i d)] / (c*c + d*d)
//   == [(a*c + b*d) / (c*c + d*d)] + i [(b*c - a*d) / (c*c + d*d)]
impl Div<Complex64> for Complex64 {
    type Output = Self;

    #[inline]
    fn div(self, other: Self) -> Self::Output {
        let norm_sqr = other.norm_sqr();
        let re = self.real.clone() * other.real + self.imag.clone() * other.imag;
        let im = self.imag * other.real - self.real * other.imag;
        Self::Output::new(re / norm_sqr.clone(), im / norm_sqr)
    }
}

impl ToComplex<Complex32> for Complex64 {
    #[inline]
    fn to_complex(&self) -> Complex32 {
        Complex32::new(self.real as f32, self.imag as f32)
    }
}

impl ToComplex<Complex64> for Complex32 {
    #[inline]
    fn to_complex(&self) -> Complex64 {
        Complex64::new(self.real as f64, self.imag as f64)
    }
}

impl ToComplex<Complex32> for Complex32 {
    #[inline]
    fn to_complex(&self) -> Complex32 {
        *self
    }
}

impl ToComplex<Complex64> for Complex64 {
    #[inline]
    fn to_complex(&self) -> Complex64 {
        *self
    }
}
#[cfg(test)]
pub(crate) mod tests {
    use burn_tensor::DType;

    use super::*;

    #[test]
    fn test_complex32_basic() {
        let c = Complex32::new(3.0, 4.0);
        assert_eq!(c.real, 3.0);
        assert_eq!(c.imag, 4.0);
        assert_eq!(c.abs(), 5.0); // 3-4-5 triangle
        assert_eq!(c.conj(), Complex32::new(3.0, -4.0));
    }

    #[test]
    fn test_complex64_basic() {
        let c = Complex64::new(3.0, 4.0);
        assert_eq!(c.real, 3.0);
        assert_eq!(c.imag, 4.0);
        assert_eq!(c.abs(), 5.0); // 3-4-5 triangle
        assert_eq!(c.conj(), Complex64::new(3.0, -4.0));
    }

    #[test]
    fn test_complex_element_traits() {
        // Test that our complex types implement Element trait
        assert_eq!(Complex32::dtype(), DType::Complex32);
        assert_eq!(Complex64::dtype(), DType::Complex64);

        // Test conversion
        let c32 = Complex32::new(1.0, 2.0);
        let c64: Complex64 = c32.to_complex();
        assert_eq!(c64.real, 1.0);
        assert_eq!(c64.imag, 2.0);
    }

    #[test]
    fn test_complex_display() {
        let c1 = Complex32::new(3.0, 4.0);
        assert_eq!(format!("{}", c1), "3+4i");

        let c2 = Complex32::new(3.0, -4.0);
        assert_eq!(format!("{}", c2), "3-4i");

        let c3 = Complex64::new(-3.0, 4.0);
        assert_eq!(format!("{}", c3), "-3+4i");
    }
}
