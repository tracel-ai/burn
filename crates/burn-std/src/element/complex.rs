//use num_complex::Complex as NumComplex;

/// 32-bit complex number type (real and imaginary parts are f32).
use crate::{
    DType, Distribution, Element, ElementComparison, ElementConversion, ElementEq, ElementRandom,
    cast::ToElement,
};
use bytemuck::Zeroable;
use core::ops::Add;
use core::ops::Div;
use core::ops::Mul;
use core::ops::Neg;
use core::ops::Sub;
use core::ops::{AddAssign, Rem};
use num_complex::Complex as NumComplex;
use num_traits::FromPrimitive;
use num_traits::Num;
use num_traits::One;
use num_traits::Zero;
use rand::Rng;
#[cfg(feature = "ndarray")]
mod ndarray {
    use super::ComplexScalar;
    use ndarray::ScalarOperand;
    impl<E: ScalarOperand> ScalarOperand for ComplexScalar<E> {}
}

#[cfg(feature = "tch")]
mod tch {
    use super::ComplexScalar;
    use tch::kind::Element as TchElement;
    // Not supported right now burn side, apparently supported for tch
    impl<E: TchElement> TchElement for ComplexScalar<f16> {
        const KIND: tch::Kind = tch::Kind::ComplexHalf;
        const ZERO: Self = Self::new(0.0, 0.0);
    }
    impl<E: TchElement> TchElement for ComplexScalar<f32> {
        const KIND: tch::Kind = tch::Kind::ComplexFloat;
        const ZERO: Self = Self::new(0.0, 0.0);
    }
    impl<E: TchElement> TchElement for ComplexScalar<f64> {
        const KIND: tch::Kind = tch::Kind::ComplexDouble;
        const ZERO: Self = Self::new(0.0, 0.0);
    }
}

/// trait to convert an element to a complex number when the conversion needs to be generic over
/// the target complex type (e.g. `Complex<f32>` or `Complex<f64>`)
pub trait ToComplex<C> {
    /// Convert self to a complex number of type C
    fn to_complex(&self) -> C;
}

/// Trait to access the real and imaginary parts of a complex element
pub trait ComplexElement: Element {
    /// The inner type of the complex number (e.g. f32 for `Complex<f32>`)
    type InnerType: Element;
    /// Get the real part of the complex number
    fn real(&self) -> Self::InnerType;
    /// Get the imaginary part of the complex number
    fn imag(&self) -> Self::InnerType;
}

impl<E: Element + ElementComparison + bytemuck::Pod> ComplexElement for ComplexScalar<E> {
    type InnerType = E;
    #[inline]
    fn real(&self) -> Self::InnerType {
        self.real
    }
    #[inline]
    fn imag(&self) -> Self::InnerType {
        self.imag
    }
}
/// Complex Element Type, essentially a copy of num_complex::Complex,
/// but with some burn specific modifications
#[derive(Clone, PartialEq)]
#[repr(C)]
pub struct ComplexScalar<E> {
    /// Real part of a complex number
    pub real: E,
    /// Imag part of a complex number
    pub imag: E,
}

impl<E> From<NumComplex<E>> for ComplexScalar<E> {
    fn from(c: NumComplex<E>) -> Self {
        Self {
            real: c.re,
            imag: c.im,
        }
    }
}

impl<E: Element + ElementComparison + bytemuck::Pod + Zero + core::ops::Add + core::iter::Sum>
    core::iter::Sum for ComplexScalar<E>
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::new(E::zero(), E::zero()), |a, b| a + b)
    }
}

impl<E> core::iter::Product for ComplexScalar<E>
where
    E: Element
        + ElementComparison
        + bytemuck::Pod
        + One
        + Zero
        + Mul<Output = E>
        + Add<Output = E>
        + Sub<Output = E>
        + Copy
        + core::iter::Product,
{
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::new(E::one(), E::zero()), |a, b| a * b)
    }
}

impl<E> From<ComplexScalar<E>> for NumComplex<E> {
    fn from(val: ComplexScalar<E>) -> Self {
        NumComplex::new(val.real, val.imag)
    }
}

impl<E: Num + core::marker::Copy> Num for ComplexScalar<E> {
    type FromStrRadixErr = <NumComplex<E> as Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        NumComplex::from_str_radix(str, radix).map(Self::from)
    }
}

impl<E: core::fmt::Debug> core::fmt::Debug for ComplexScalar<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Complex {{ real: {:?}, imag: {:?} }}",
            self.real, self.imag
        )
    }
}

impl<E: Copy> Copy for ComplexScalar<E> {}

unsafe impl<E: bytemuck::Zeroable> Zeroable for ComplexScalar<E> {}

unsafe impl<E: bytemuck::Pod> bytemuck::Pod for ComplexScalar<E> {}

impl<E> Default for ComplexScalar<E>
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

impl<E: core::fmt::Display + Element + ElementComparison> core::fmt::Display for ComplexScalar<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.imag.cmp(&E::zeroed()).is_ge() {
            write!(f, "{}+{}i", self.real, self.imag)
        } else {
            write!(f, "{}{}i", self.real, self.imag)
        }
    }
}

impl<E> AddAssign for ComplexScalar<E>
where
    E: AddAssign,
{
    fn add_assign(&mut self, rhs: Self) {
        self.real += rhs.real;
        self.imag += rhs.imag;
    }
}

impl<E> Add for ComplexScalar<E>
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

impl<E: Neg<Output = E>> Neg for ComplexScalar<E> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self {
            real: -self.real,
            imag: -self.imag,
        }
    }
}

impl<E: Sub<Output = E>> Sub for ComplexScalar<E> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            real: self.real - rhs.real,
            imag: self.imag - rhs.imag,
        }
    }
}

impl<E: Mul<Output = E> + Add<Output = E> + Sub<Output = E> + Copy> Mul for ComplexScalar<E> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            real: self.real * rhs.real - self.imag * rhs.imag,
            imag: self.real * rhs.imag + self.imag * rhs.real,
        }
    }
}

impl<E: Rem<Output = E>> Rem for ComplexScalar<E> {
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
impl<E> Div for ComplexScalar<E>
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

impl<E> ToElement for ComplexScalar<E>
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

    #[inline]
    fn to_complex32(&self) -> ComplexScalar<f32> {
        ComplexScalar::<f32>::new(self.real.to_f32(), self.imag.to_f32())
    }
    #[inline]
    fn to_complex64(&self) -> ComplexScalar<f64> {
        ComplexScalar::<f64>::new(self.real.to_f64(), self.imag.to_f64())
    }
}

impl<E: Neg<Output = E>> ComplexScalar<E> {
    /// Get the conjugate of the complex number
    #[inline]
    pub fn conj(self) -> Self {
        Self {
            real: self.real,
            imag: -self.imag,
        }
    }
}

impl<E> FromPrimitive for ComplexScalar<E>
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

impl<C> ElementConversion for ComplexScalar<C>
where
    C: Element + ToElement,
{
    #[inline(always)]
    fn from_elem<E: ToElement>(elem: E) -> Self {
        ComplexScalar::<C> {
            real: C::from_elem(elem),
            imag: C::from_elem(0.0),
        }
    }
    #[inline(always)]
    fn elem<E: Element>(self) -> E {
        E::from_elem(self)
    }
}

impl<E: Zero> Zero for ComplexScalar<E> {
    fn zero() -> Self {
        Self::new(E::zero(), E::zero())
    }
    fn is_zero(&self) -> bool {
        self.real.is_zero() && self.imag.is_zero()
    }
}

impl<E> One for ComplexScalar<E>
where
    E: One + Zero + Mul<Output = E> + Copy + Sub<Output = E>,
{
    #[inline]
    fn one() -> Self {
        Self::new(E::one(), E::zero())
    }
}

impl<E> ComplexScalar<E> {
    /// Create a new complex number from real and imaginary parts
    #[inline]
    pub fn new(real: E, imag: E) -> Self {
        Self { real, imag }
    }
    /// Get the real part of the complex number
    #[inline]
    pub fn real(self) -> E {
        self.real
    }
    /// Get the imaginary part of the complex number
    #[inline]
    pub fn imag(self) -> E {
        self.imag
    }
}

impl<E> ComplexScalar<E>
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

        let ComplexScalar { real, mut imag } = self;
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
    /// See also \[`Complex::i`\].
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
        if exp == ComplexScalar::<E>::new(E::zero(), E::zero()) {
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

    /// Computes the principal value of the sine of `self`.
    pub fn sin(self) -> Self {
        // formula: sin(a + bi) = sin(a) cosh(b) + i cos(a) sinh(b)
        Self::new(
            self.real.sin() * self.imag.cosh(),
            self.real.cos() * self.imag.sinh(),
        )
    }
    /// Computes the principal value of the tangent of `self`.
    pub fn tan(self) -> Self {
        // formula: tan(z) = sin(z) / cos(z)
        self.sin() / self.cos()
    }
    /// Computes the principal value of the cosine of `self`.
    pub fn cos(self) -> Self {
        // formula: cos(a + bi) = cos(a) cosh(b) - i sin(a) sinh(b)
        Self::new(
            self.real.cos() * self.imag.cosh(),
            -self.real.sin() * self.imag.sinh(),
        )
    }

    /// Get the magnitude (absolute value) of the complex number
    #[inline]
    pub fn abs(self) -> E {
        (self.real * self.real + self.imag * self.imag).sqrt()
    }

    /// Computes the principal value of the inverse sine of `self`.
    ///
    /// This function has two branch cuts:
    ///
    /// * `(-∞, -1)`, continuous from above.
    /// * `(1, ∞)`, continuous from below.
    ///
    /// The branch satisfies `-π/2 ≤ Re(asin(z)) ≤ π/2`.
    #[inline]
    pub fn asin(self) -> Self {
        // formula: arcsin(z) = -i ln(sqrt(1-z^2) + i*z)
        let i = Self::i();
        -i * ((Self::one() - self * self).sqrt() + i * self).ln()
    }

    /// Computes the principal value of the inverse cosine of `self`.
    ///
    /// This function has two branch cuts:
    ///
    /// * `(-∞, -1)`, continuous from above.
    /// * `(1, ∞)`, continuous from below.
    ///
    /// The branch satisfies `0 ≤ Re(acos(z)) ≤ π`.
    #[inline]
    pub fn acos(self) -> Self {
        // formula: arccos(z) = -i ln(i sqrt(1-z^2) + z)
        let i = Self::i();
        -i * (i * (Self::one() - self * self).sqrt() + self).ln()
    }

    /// Computes the hyperbolic sine of `self`.
    #[inline]
    pub fn sinh(self) -> Self {
        // formula: sinh(a + bi) = sinh(a)cos(b) + i*cosh(a)sin(b)
        Self::new(
            self.real.sinh() * self.imag.cos(),
            self.real.cosh() * self.imag.sin(),
        )
    }

    /// Computes the hyperbolic cosine of `self`.
    #[inline]
    pub fn cosh(self) -> Self {
        // formula: cosh(a + bi) = cosh(a)cos(b) + i*sinh(a)sin(b)
        Self::new(
            self.real.cosh() * self.imag.cos(),
            self.real.sinh() * self.imag.sin(),
        )
    }

    /// Computes the hyperbolic tangent of `self`.
    #[inline]
    pub fn tanh(self) -> Self {
        // formula: tanh(a + bi) = (sinh(2a) + i*sin(2b))/(cosh(2a) + cos(2b))
        let (two_real, two_imag) = (self.real + self.real, self.imag + self.imag);
        Self::new(two_real.sinh(), two_imag.sin()).unscale(two_real.cosh() + two_imag.cos())
    }

    /// Divides `self` by the scalar `t`.
    #[inline]
    pub fn unscale(&self, t: E) -> Self {
        Self::new(self.real / t, self.imag / t)
    }

    /// Computes the principal value of inverse hyperbolic sine of `self`.
    ///
    /// This function has two branch cuts:
    ///
    /// * `(-∞i, -i)`, continuous from the left.
    /// * `(i, ∞i)`, continuous from the right.
    ///
    /// The branch satisfies `-π/2 ≤ Im(asinh(z)) ≤ π/2`.
    #[inline]
    pub fn asinh(self) -> Self {
        // formula: arcsinh(z) = ln(z + sqrt(1+z^2))
        let one = Self::one();
        (self + (one + self * self).sqrt()).ln()
    }

    /// Computes the principal value of inverse hyperbolic cosine of `self`.
    ///
    /// This function has one branch cut:
    ///
    /// * `(-∞, 1)`, continuous from above.
    ///
    /// The branch satisfies `-π ≤ Im(acosh(z)) ≤ π` and `0 ≤ Re(acosh(z)) < ∞`.
    #[inline]
    pub fn acosh(self) -> Self {
        // formula: arccosh(z) = 2 ln(sqrt((z+1)/2) + sqrt((z-1)/2))
        let one = Self::one();
        let two = one + one;
        two * (((self + one) / two).sqrt() + ((self - one) / two).sqrt()).ln()
    }

    /// Computes the principal value of inverse hyperbolic tangent of `self`.
    ///
    /// This function has two branch cuts:
    ///
    /// * `(-∞, -1]`, continuous from above.
    /// * `[1, ∞)`, continuous from below.
    ///
    /// The branch satisfies `-π/2 ≤ Im(atanh(z)) ≤ π/2`.
    #[inline]
    pub fn atanh(self) -> Self {
        // formula: arctanh(z) = (ln(1+z) - ln(1-z))/2
        let one = Self::one();
        let two = one + one;
        if self == one {
            return Self::new(E::infinity(), E::zero());
        } else if self == -one {
            return Self::new(-E::infinity(), E::zero());
        }
        ((one + self).ln() - (one - self).ln()) / two
    }
}

impl<E: Element + ElementComparison + bytemuck::Pod> Element for ComplexScalar<E> {
    #[inline(always)]
    fn dtype() -> DType {
        match E::dtype() {
            DType::F32 => DType::Complex32,
            DType::F64 => DType::Complex64,
            _ => panic!("Unsupported element type for Complex. Only f32 and f64 are supported."),
        }
    }
}

impl<E> ComplexScalar<E>
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

impl<E: ElementRandom> ElementRandom for ComplexScalar<E> {
    fn random<R: Rng>(distribution: Distribution, rng: &mut R) -> Self {
        ComplexScalar::<E>::new(E::random(distribution, rng), E::random(distribution, rng))
    }
}

impl<E: ElementEq> ElementEq for ComplexScalar<E> {
    fn eq(&self, other: &Self) -> bool {
        self.real.eq(&other.real) && self.imag.eq(&other.imag)
    }
}

macro_rules! to_complex {
    (
        $type:ident
    ) => {
        impl ToComplex<ComplexScalar<f32>> for $type {
            #[inline]
            fn to_complex(&self) -> ComplexScalar<f32> {
                ComplexScalar::<f32>::new(self.to_f32(), 0.0)
            }
        }
        impl ToComplex<ComplexScalar<f64>> for $type {
            #[inline]
            fn to_complex(&self) -> ComplexScalar<f64> {
                ComplexScalar::<f64>::new(self.to_f64(), 0.0)
            }
        }
    };
}

to_complex!(i64);
to_complex!(i32);
to_complex!(f32);
to_complex!(f64);
to_complex!(u64);
to_complex!(u32);
to_complex!(u16);
to_complex!(u8);
to_complex!(bool);

impl ToComplex<ComplexScalar<f32>> for ComplexScalar<f64> {
    #[inline]
    fn to_complex(&self) -> ComplexScalar<f32> {
        ComplexScalar::<f32>::new(self.real as f32, self.imag as f32)
    }
}

impl ToComplex<ComplexScalar<f64>> for ComplexScalar<f32> {
    #[inline]
    fn to_complex(&self) -> ComplexScalar<f64> {
        ComplexScalar::<f64>::new(self.real as f64, self.imag as f64)
    }
}

impl ToComplex<ComplexScalar<f32>> for ComplexScalar<f32> {
    #[inline]
    fn to_complex(&self) -> ComplexScalar<f32> {
        *self
    }
}

impl ToComplex<ComplexScalar<f64>> for ComplexScalar<f64> {
    #[inline]
    fn to_complex(&self) -> ComplexScalar<f64> {
        *self
    }
}
#[cfg(test)]
pub(crate) mod tests {

    use super::*;
    extern crate alloc;

    #[test]
    fn test_complex32_basic() {
        let c = ComplexScalar::<f32>::new(3.0, 4.0);
        assert_eq!(c.real, 3.0);
        assert_eq!(c.imag, 4.0);
        assert_eq!(c.abs(), 5.0); // 3-4-5 triangle
        assert_eq!(c.conj(), ComplexScalar::<f32>::new(3.0, -4.0));
    }

    #[test]
    fn test_complex64_basic() {
        let c = ComplexScalar::<f64>::new(3.0, 4.0);
        assert_eq!(c.real, 3.0);
        assert_eq!(c.imag, 4.0);
        assert_eq!(c.abs(), 5.0); // 3-4-5 triangle
        assert_eq!(c.conj(), ComplexScalar::<f64>::new(3.0, -4.0));
    }

    #[test]
    fn test_complex_element_traits() {
        // Test that our complex types implement Element trait
        assert_eq!(ComplexScalar::<f32>::dtype(), DType::Complex32);
        assert_eq!(ComplexScalar::<f64>::dtype(), DType::Complex64);

        // Test conversion
        let c32 = ComplexScalar::<f32>::new(1.0, 2.0);
        let c64: ComplexScalar<f64> = c32.to_complex();
        assert_eq!(c64.real, 1.0);
        assert_eq!(c64.imag, 2.0);
    }

    #[test]
    fn test_complex_display() {
        let c1 = ComplexScalar::<f32>::new(3.0, 4.0);
        assert_eq!(alloc::format!("{}", c1), "3+4i");

        let c2 = ComplexScalar::<f32>::new(3.0, -4.0);
        assert_eq!(alloc::format!("{}", c2), "3-4i");

        let c3 = ComplexScalar::<f64>::new(-3.0, 4.0);
        assert_eq!(alloc::format!("{}", c3), "-3+4i");
    }
}
