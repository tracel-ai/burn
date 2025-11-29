//use num_complex::Complex as NumComplex;

/// 32-bit complex number type (real and imaginary parts are f32).
use burn_tensor::{
    DType, Distribution, Element, ElementComparison, ElementConversion, ElementLimits,
    ElementRandom, cast::ToElement,
};
use core::cmp::Ordering;
use num_traits::identities::ConstZero;
use rand::RngCore;
pub trait ToComplex<C> {
    fn to_complex(&self) -> C;
}
use paste::paste;
pub trait ToComplexElement: ToElement + ToComplex<Complex32> + ToComplex<Complex64> {
    fn to_complex32(&self) -> Complex32 {
        self.to_complex()
    }
    fn to_complex64(&self) -> Complex64 {
        self.to_complex()
    }
}

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
        }

        impl ToComplex<Complex32> for $type {
            fn to_complex(&self) -> Complex32 {
                Complex32::new(self.real as f32, self.imag as f32)
            }
        }

        impl ToComplex<Complex64> for $type {
            fn to_complex(&self) -> Complex64 {
                Complex64::new(self.real as f64, self.imag as f64)
            }
        }

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
            fn random<R: RngCore>(distribution: Distribution, rng: &mut R) -> Self {
                #[allow(clippy::redundant_closure_call)]
                $random(distribution, rng)
            }
        }

        impl ElementComparison for $type {
            fn cmp(&self, other: &Self) -> Ordering {
                let a = self.to_complex();
                let b = other.to_complex();
                #[allow(clippy::redundant_closure_call)]
                $cmp(&a, &b)
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

        impl Element for $type {
            fn dtype() -> DType {
                $dtype
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
