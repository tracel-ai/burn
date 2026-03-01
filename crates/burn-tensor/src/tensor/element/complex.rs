use core::cmp::Ordering;

use crate::{
    DType, Distribution, Element, ElementComparison, ElementConversion, ElementLimits,
    ElementPrecision, ElementRandom, Precision, cast::ToElement, make_element,
};
use num_traits::ConstZero;
use rand::RngCore;

macro_rules! make_complex {
    (
        ty $type:ident $inner:ident $precision:expr,
        convert $convert:expr,
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
        /// Complex number with real and imaginary parts of type `$inner`.
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

            #[inline]
            fn to_complex32(&self) -> Complex32 {
                Complex32::new(self.real as f32, self.imag as f32)
            }
            #[inline]
            fn to_complex64(&self) -> Complex64 {
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

        make_element!(
            ty $type $precision,
            convert $convert,
            random $random,
            cmp $cmp, dtype $dtype, min $min, max $max);
    }
}

make_complex!(
    ty Complex32 f32 Precision::Full,
    convert ToElement::to_complex32,
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

make_complex!(
    ty Complex64 f64 Precision::Double,
    convert ToElement::to_complex64,
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
