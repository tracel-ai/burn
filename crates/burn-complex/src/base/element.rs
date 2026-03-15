//use num_complex::Complex as NumComplex;

use burn_tensor::ElementOrdered;
/// 32-bit complex number type (real and imaginary parts are f32).
use burn_tensor::{
    DType, Distribution, Element, ElementComparison, ElementConversion, ElementEq, ElementLimits,
    ElementRandom, cast::ToElement,
};
use bytemuck::Zeroable;
use rand::seq::index;

use core::ops::{AddAssign, Rem};
use num_traits::FromPrimitive;
use num_traits::Num;
use num_traits::One;
use num_traits::Pow;
use num_traits::Zero;
use num_traits::float::FloatCore;
use num_traits::identities::ConstZero;
use rand::Rng;
use std::ops::Add;
use std::ops::Mul;
use std::ops::Neg;
use std::ops::Sub;
#[cfg(feature = "ndarray")]
mod ndarray {
    use super::Complex;
    use ndarray::ScalarOperand;
    impl<E: ScalarOperand> ScalarOperand for Complex<E> {}
}

#[cfg(feature = "tch")]
mod tch {
    use super::Complex;
    use tch::kind::Element as TchElement;
    impl<E: TchElement> TchElement for Complex<E> {}
}

use std::ops::Div;
pub trait ToComplex<C> {
    fn to_complex(&self) -> C;
}

use paste::paste;

use crate::base::Complex as ComplexTensor;
use crate::base::ComplexTensorOps;
pub trait ToComplexElement: ToElement {
    fn to_complex32(&self) -> Complex<f32>;
    fn to_complex64(&self) -> Complex<f64>;
}

// will attempt after I get ndarray to compile

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Complex<E> {
    pub real: E,
    pub imag: E,
}

unsafe impl<E: bytemuck::Zeroable> Zeroable for Complex<E> {}

unsafe impl<E: bytemuck::Pod> bytemuck::Pod for Complex<E> {}

impl<E> Default for Complex<E>
where
    E: Default,
{
    fn default() -> Self {
        Self {
            real: E::default(),
            imag: E::default(),
        }
    }
}

impl<E: core::fmt::Display + Element + ElementComparison> core::fmt::Display for Complex<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.imag.cmp(&E::zeroed()).is_ge() {
            write!(f, "{}+{}i", self.real, self.imag)
        } else {
            write!(f, "{}{}i", self.real, self.imag)
        }
    }
}

impl<E> AddAssign for Complex<E>
where
    E: AddAssign,
{
    fn add_assign(&mut self, rhs: Self) {
        self.real += rhs.real;
        self.imag += rhs.imag;
    }
}

impl<E> Add for Complex<E>
where
    E: Add<Output = E>,
{
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            real: self.real + rhs.real,
            imag: self.imag + rhs.imag,
        }
    }
}

impl<E: Neg<Output = E>> Neg for Complex<E> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self {
            real: -self.real,
            imag: -self.imag,
        }
    }
}

impl<E: Sub<Output = E>> Sub for Complex<E> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            real: self.real - rhs.real,
            imag: self.imag - rhs.imag,
        }
    }
}

impl<E: Mul<Output = E> + Add<Output = E> + Sub<Output = E> + Copy> Mul for Complex<E> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            real: self.real * rhs.real - self.imag * rhs.imag,
            imag: self.real * rhs.imag + self.imag * rhs.real,
        }
    }
}

impl<E: Rem<Output = E>> Rem for Complex<E> {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        Self {
            real: self.real % rhs.real,
            imag: self.imag % rhs.imag,
        }
    }
}

// (a + i b) / (c + i d) == [(a + i b) * (c - i d)] / (c*c + d*d)
//   == [(a*c + b*d) / (c*c + d*d)] + i [(b*c - a*d) / (c*c + d*d)]
impl<E> Div for Complex<E>
where
    E: Mul<Output = E> + Add<Output = E> + Div<Output = E> + Sub<Output = E> + Copy,
{
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        let norm_sqr = rhs.real * rhs.real + rhs.imag * rhs.imag;
        let re = self.real * rhs.real + self.imag * rhs.imag;
        let im = self.imag * rhs.real - self.real * rhs.imag;
        Self::Output::new(re / norm_sqr, im / norm_sqr)
    }
}

impl<E> ToElement for Complex<E>
where
    E: ToElement,
{
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
        self.real.to_f32()
    }
    #[inline]
    fn to_f64(&self) -> f64 {
        self.real.to_f64()
    }
    #[inline]
    fn to_bool(&self) -> bool {
        self.real.to_bool() || self.imag.to_bool()
    }
}

impl<E: Neg<Output = E>> Complex<E> {
    /// Get the conjugate of the complex number
    #[inline]
    pub fn conj(self) -> Self {
        Self {
            real: self.real,
            imag: -self.imag,
        }
    }
}

impl<E> FromPrimitive for Complex<E>
where
    E: FromPrimitive + num_traits::identities::Zero,
{
    fn from_i64(n: i64) -> Option<Self> {
        E::from_i64(n).map(Self::from_real)
    }
    fn from_u64(n: u64) -> Option<Self> {
        E::from_u64(n).map(Self::from_real)
    }
    fn from_f32(n: f32) -> Option<Self> {
        E::from_f32(n).map(Self::from_real)
    }
    fn from_f64(n: f64) -> Option<Self> {
        E::from_f64(n).map(Self::from_real)
    }
}

impl<C> ElementConversion for Complex<C>
where
    C: Element + ToElement,
{
    #[inline(always)]
    fn from_elem<E: ToElement>(elem: E) -> Self {
        Complex::<C> {
            real: C::from_elem(elem),
            imag: C::from_elem(0.0),
        }
    }
    #[inline(always)]
    fn elem<E: Element>(self) -> E {
        E::from_elem(self)
    }
}

impl<E: Zero> Zero for Complex<E> {
    fn zero() -> Self {
        Self::new(E::zero(), E::zero())
    }
    fn is_zero(&self) -> bool {
        self.real.is_zero() && self.imag.is_zero()
    }
}

impl<E> One for Complex<E>
where
    E: One + Zero + Mul<Output = E> + Copy + Sub<Output = E>,
{
    #[inline]
    fn one() -> Self {
        Self::new(E::one(), E::zero())
    }
}

impl<E> Complex<E> {
    /// Create a new complex number from real and imaginary parts
    #[inline]
    pub fn new(real: E, imag: E) -> Self {
        Self { real, imag }
    }
    #[inline]
    pub fn real(self) -> E {
        self.real
    }
    #[inline]
    pub fn imag(self) -> E {
        self.imag
    }
}

impl<E> Complex<E>
where
    E: num_traits::Float,
{
    // The below methods are copied from num_complex 0.4.6, since we can't implement the required element traits for num_complex::Complex.
    // link to the docs: https://docs.rs/num-complex/0.4.6/num_complex/
    // Credit to https://github.com/cuviper for the original implementations.

    /// Computes `e^(self)`, where `e` is the base of the natural logarithm.
    #[inline]
    pub fn exp(self) -> Self {
        // formula: e^(a + bi) = e^a (cos(b) + i*sin(b)) = from_polar(e^a, b)

        let Complex { real, mut imag } = self;
        // Treat the corner cases +∞, -∞, and NaN
        if real.is_infinite() {
            if real < E::zero() {
                if !imag.is_finite() {
                    return Self::new(E::zero(), E::zero());
                }
            } else if imag == E::zero() || !imag.is_finite() {
                if imag.is_infinite() {
                    imag = E::nan();
                }
                return Self::new(real, imag);
            }
        } else if real.is_nan() && imag == E::zero() {
            return self;
        }

        Self::from_polar(real.exp(), imag)
    }
    /// Convert a polar representation into a complex number.
    #[inline]
    pub fn from_polar(r: E, theta: E) -> Self {
        Self::new(r * theta.cos(), r * theta.sin())
    }

    /// Calculate |self|
    #[inline]
    pub fn norm(self) -> E {
        self.real.hypot(self.imag)
    }

    /// Convert to polar form (r, theta), such that
    /// `self = r * exp(i * theta)`
    #[inline]
    pub fn to_polar(self) -> (E, E) {
        (self.norm(), self.arg())
    }

    /// Calculate the principal Arg of self.
    #[inline]
    pub fn arg(self) -> E {
        self.imag.atan2(self.real)
    }

    /// Returns the logarithm of `self` with respect to an arbitrary base.
    #[inline]
    pub fn log(self, base: E) -> Self {
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
            return Self::new(E::zero(), E::infinity());
        } else if self == -i {
            return Self::new(E::zero(), -E::infinity());
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

    /// Returns the imaginary unit.
    ///
    /// See also [`Complex::I`].
    #[inline]
    pub fn i() -> Self {
        Self::new(E::zero(), E::one())
    }

    /// Returns the square of the norm (since `T` doesn't necessarily
    /// have a sqrt function), i.e. `re^2 + im^2`.
    #[inline]
    pub fn norm_sqr(&self) -> E {
        self.real * self.real + self.imag * self.imag
    }

    /// Raises `self` to a floating point power.
    #[inline]
    pub fn powf(self, exp: E) -> Self {
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
        if exp == Complex::<E>::new(E::zero(), E::zero()) {
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
                let re = E::zero();
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
            let one = E::one();
            let two = one + one;
            let x = (self.imag.abs() / two).sqrt();
            if self.imag.is_sign_positive() {
                Self::new(x, x)
            } else {
                Self::new(x, -x)
            }
        } else {
            // formula: sqrt(r e^(it)) = sqrt(r) e^(it/2)
            let one = E::one();
            let two = one + one;
            let (r, theta) = self.to_polar();
            Self::from_polar(r.sqrt(), theta / two)
        }
    }
    /// Get the magnitude (absolute value) of the complex number
    #[inline]
    pub fn abs(self) -> E {
        (self.real * self.real + self.imag * self.imag).sqrt()
    }
}

impl<E> Complex<E>
where
    E: num_traits::identities::Zero,
{
    /// Create a complex number from a real number
    #[inline]
    pub fn from_real(real: E) -> Self {
        Self {
            real,
            imag: E::zero(),
        }
    }
}

impl<E: ElementRandom> ElementRandom for Complex<E> {
    fn random<R: Rng>(distribution: Distribution, rng: &mut R) -> Self {
        Complex::<E>::new(E::random(distribution, rng), E::random(distribution, rng))
    }
}

impl<E: ElementLimits> ElementLimits for Complex<E> {
    const MIN: Self = Complex::<E> {
        real: E::MIN,
        imag: E::MIN,
    };
    const MAX: Self = Complex::<E> {
        real: E::MAX,
        imag: E::MAX,
    };
}

impl<E: ToElement> ToComplexElement for Complex<E> {
    #[inline]
    fn to_complex32(&self) -> Complex<f32> {
        Complex::<f32>::new(self.real.to_f32(), self.imag.to_f32())
    }
    #[inline]
    fn to_complex64(&self) -> Complex<f64> {
        Complex::<f64>::new(self.real.to_f64(), self.imag.to_f64())
    }
}

impl<E: ElementEq> ElementEq for Complex<E> {
    fn eq(&self, other: &Self) -> bool {
        self.real.eq(&other.real) && self.imag.eq(&other.imag)
    }
}

impl<E: ElementOrdered + bytemuck::Pod> Element for Complex<E> {
    #[inline]
    fn dtype() -> DType {
        match E::dtype() {
            DType::F32 => DType::Complex32,
            DType::F64 => DType::Complex64,
            _ => panic!("Unsupported element type for Complex: {:?}", E::dtype()),
        }
    }
}

/// Macro to implement the element trait for a type.
#[macro_export]
macro_rules! make_complex {
    (
        $inner:ident $precision:expr,
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


        impl Complex<$inner> {





            /// Create a complex number from any element primitive
            #[inline]
            pub fn from_elem<E: ToElement>(real: E) -> Self {
                paste! {
                    Self { real: real.[<to_ $inner>](), imag: $inner::ZERO }
                }
            }






        }







        // impl<Complex<$inner>> ToComplex<Complex32> for Complex<$inner> {
        //     fn to_complex(&self) -> Complex32 {
        //         Complex32::new(self.real as f32, self.imag as f32)
        //     }
        // }

        // impl ToComplex<Complex64> for Complex<$inner> {
        //     fn to_complex(&self) -> Complex64 {
        //         Complex64::new(self.real as f64, self.imag as f64)
        //     }
        // }


















    };
}

make_complex!(
    ty Complex32 f32 Precision::Full,
    convert ToComplexElement::to_complex32,
    random |distribution: Distribution, rng: &mut R| {
        let real: f32 = distribution.sampler(rng).sample();
        let imag: f32 = distribution.sampler(rng).sample();
        Complex::<f32>::new(real, imag)
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
    min Complex::<f32>{real: f32::MIN, imag: f32::MIN},
    max Complex::<f32>{real: f32::MAX, imag: f32::MAX}
);
macro_rules! to_complex {
    (
        $type:ident
    ) => {
        impl ToComplex<Complex<f32>> for $type {
            #[inline]
            fn to_complex(&self) -> Complex<f32> {
                Complex::<f32>::new(*self as f32, 0.0)
            }
        }
        impl ToComplex<Complex<f64>> for $type {
            #[inline]
            fn to_complex(&self) -> Complex<f64> {
                Complex::<f64>::new(*self as f64, 0.0)
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
        Complex::<f64>::new(real, imag)
    },
    cmp |a: &Complex<f64>, b: &Complex<f64>| {
        // Compare by magnitude, then by real part if magnitudes are equal
        let mag_cmp = a.abs().total_cmp(&b.abs());
        if mag_cmp == Ordering::Equal {
            a.real.total_cmp(&b.real)
        } else {
            mag_cmp
        }
    },
    dtype DType::Complex64,
    min Complex::<f64>{real: f64::MIN, imag: f64::MIN},
    max Complex::<f64>{real: f64::MAX, imag: f64::MAX}


);

impl ToComplex<Complex<f32>> for Complex<f64> {
    #[inline]
    fn to_complex(&self) -> Complex<f32> {
        Complex::<f32>::new(self.real as f32, self.imag as f32)
    }
}

impl ToComplex<Complex<f64>> for Complex<f32> {
    #[inline]
    fn to_complex(&self) -> Complex<f64> {
        Complex::<f64>::new(self.real as f64, self.imag as f64)
    }
}

impl ToComplex<Complex<f32>> for Complex<f32> {
    #[inline]
    fn to_complex(&self) -> Complex<f32> {
        *self
    }
}

impl ToComplex<Complex<f64>> for Complex<f64> {
    #[inline]
    fn to_complex(&self) -> Complex<f64> {
        *self
    }
}
#[cfg(test)]
pub(crate) mod tests {
    use burn_tensor::DType;

    use super::*;

    #[test]
    fn test_complex32_basic() {
        let c = Complex::<f32>::new(3.0, 4.0);
        assert_eq!(c.real, 3.0);
        assert_eq!(c.imag, 4.0);
        assert_eq!(c.abs(), 5.0); // 3-4-5 triangle
        assert_eq!(c.conj(), Complex::<f32>::new(3.0, -4.0));
    }

    #[test]
    fn test_complex64_basic() {
        let c = Complex::<f64>::new(3.0, 4.0);
        assert_eq!(c.real, 3.0);
        assert_eq!(c.imag, 4.0);
        assert_eq!(c.abs(), 5.0); // 3-4-5 triangle
        assert_eq!(c.conj(), Complex::<f64>::new(3.0, -4.0));
    }

    #[test]
    fn test_complex_element_traits() {
        // Test that our complex types implement Element trait
        assert_eq!(Complex::<f32>::dtype(), DType::Complex32);
        assert_eq!(Complex::<f64>::dtype(), DType::Complex64);

        // Test conversion
        let c32 = Complex::<f32>::new(1.0, 2.0);
        let c64: Complex<f64> = c32.to_complex();
        assert_eq!(c64.real, 1.0);
        assert_eq!(c64.imag, 2.0);
    }

    #[test]
    fn test_complex_display() {
        let c1 = Complex::<f32>::new(3.0, 4.0);
        assert_eq!(format!("{}", c1), "3+4i");

        let c2 = Complex::<f32>::new(3.0, -4.0);
        assert_eq!(format!("{}", c2), "3-4i");

        let c3 = Complex::<f64>::new(-3.0, 4.0);
        assert_eq!(format!("{}", c3), "-3+4i");
    }
}
