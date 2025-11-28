use core::cmp::Ordering;
use rand::RngCore;

use crate::distribution::Distribution;
use burn_std::{DType, bf16, f16};

#[cfg(feature = "cubecl")]
use burn_std::flex32;

use super::cast::ToElement;

/// Element trait for tensor.
pub trait Element:
    ToElement
    + ElementRandom
    + ElementConversion
    + ElementComparison
    + ElementLimits
    + bytemuck::CheckedBitPattern
    + bytemuck::NoUninit
    + bytemuck::Zeroable
    + core::fmt::Debug
    + core::fmt::Display
    + Default
    + Send
    + Sync
    + Copy
    + 'static
{
    /// The dtype of the element.
    fn dtype() -> DType;
}

/// Element conversion trait for tensor.
pub trait ElementConversion {
    /// Converts an element to another element.
    ///
    /// # Arguments
    ///
    /// * `elem` - The element to convert.
    ///
    /// # Returns
    ///
    /// The converted element.
    fn from_elem<E: ToElement>(elem: E) -> Self;

    /// Converts and returns the converted element.
    fn elem<E: Element>(self) -> E;
}

/// Element trait for random value of a tensor.
pub trait ElementRandom {
    /// Returns a random value for the given distribution.
    ///
    /// # Arguments
    ///
    /// * `distribution` - The distribution to sample from.
    /// * `rng` - The random number generator.
    ///
    /// # Returns
    ///
    /// The random value.
    fn random<R: RngCore>(distribution: Distribution, rng: &mut R) -> Self;
}

/// Element ordering trait.
pub trait ElementComparison {
    /// Returns and [Ordering] between `self` and `other`.
    fn cmp(&self, other: &Self) -> Ordering;
}

/// Element ordering trait.
pub trait ElementLimits {
    /// The minimum representable value
    const MIN: Self;
    /// The maximum representable value
    const MAX: Self;
}

/// Macro to implement the element trait for a type.
#[macro_export]
macro_rules! make_element {
    (
        ty $type:ident,
        convert $convert:expr,
        random $random:expr,
        cmp $cmp:expr,
        dtype $dtype:expr
    ) => {
        make_element!(ty $type, convert $convert, random $random, cmp $cmp, dtype $dtype, min $type::MIN, max $type::MAX);
    };
    (
        ty $type:ident,
        convert $convert:expr,
        random $random:expr,
        cmp $cmp:expr,
        dtype $dtype:expr,
        min $min:expr,
        max $max:expr
    ) => {
        impl Element for $type {
            #[inline(always)]
            fn dtype() -> burn_std::DType {
                $dtype
            }
        }

        impl ElementConversion for $type {
            #[inline(always)]
            fn from_elem<E: ToElement>(elem: E) -> Self {
                #[allow(clippy::redundant_closure_call)]
                $convert(&elem)
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
                let a = self.elem::<$type>();
                let b = other.elem::<$type>();
                #[allow(clippy::redundant_closure_call)]
                $cmp(&a, &b)
            }
        }

        impl ElementLimits for $type {
            const MIN: Self = $min;
            const MAX: Self = $max;
        }
    };
}

make_element!(
    ty f64,
    convert ToElement::to_f64,
    random |distribution: Distribution, rng: &mut R| distribution.sampler(rng).sample(),
    cmp |a: &f64, b: &f64| a.total_cmp(b),
    dtype DType::F64
);

make_element!(
    ty f32,
    convert ToElement::to_f32,
    random |distribution: Distribution, rng: &mut R| distribution.sampler(rng).sample(),
    cmp |a: &f32, b: &f32| a.total_cmp(b),
    dtype DType::F32
);

make_element!(
    ty i64,
    convert ToElement::to_i64,
    random |distribution: Distribution, rng: &mut R| distribution.sampler(rng).sample(),
    cmp |a: &i64, b: &i64| Ord::cmp(a, b),
    dtype DType::I64
);

make_element!(
    ty u64,
    convert ToElement::to_u64,
    random |distribution: Distribution, rng: &mut R| distribution.sampler(rng).sample(),
    cmp |a: &u64, b: &u64| Ord::cmp(a, b),
    dtype DType::U64
);

make_element!(
    ty i32,
    convert ToElement::to_i32,
    random |distribution: Distribution, rng: &mut R| distribution.sampler(rng).sample(),
    cmp |a: &i32, b: &i32| Ord::cmp(a, b),
    dtype DType::I32
);

make_element!(
    ty u32,
    convert ToElement::to_u32,
    random |distribution: Distribution, rng: &mut R| distribution.sampler(rng).sample(),
    cmp |a: &u32, b: &u32| Ord::cmp(a, b),
    dtype DType::U32
);

make_element!(
    ty i16,
    convert ToElement::to_i16,
    random |distribution: Distribution, rng: &mut R| distribution.sampler(rng).sample(),
    cmp |a: &i16, b: &i16| Ord::cmp(a, b),
    dtype DType::I16
);

make_element!(
    ty u16,
    convert ToElement::to_u16,
    random |distribution: Distribution, rng: &mut R| distribution.sampler(rng).sample(),
    cmp |a: &u16, b: &u16| Ord::cmp(a, b),
    dtype DType::U16
);

make_element!(
    ty i8,
    convert ToElement::to_i8,
    random |distribution: Distribution, rng: &mut R| distribution.sampler(rng).sample(),
    cmp |a: &i8, b: &i8| Ord::cmp(a, b),
    dtype DType::I8
);

make_element!(
    ty u8,
    convert ToElement::to_u8,
    random |distribution: Distribution, rng: &mut R| distribution.sampler(rng).sample(),
    cmp |a: &u8, b: &u8| Ord::cmp(a, b),
    dtype DType::U8
);

make_element!(
    ty f16,
    convert ToElement::to_f16,
    random |distribution: Distribution, rng: &mut R| {
        let sample: f32 = distribution.sampler(rng).sample();
        f16::from_elem(sample)
    },
    cmp |a: &f16, b: &f16| a.total_cmp(b),
    dtype DType::F16
);
make_element!(
    ty bf16,
    convert ToElement::to_bf16,
    random |distribution: Distribution, rng: &mut R| {
        let sample: f32 = distribution.sampler(rng).sample();
        bf16::from_elem(sample)
    },
    cmp |a: &bf16, b: &bf16| a.total_cmp(b),
    dtype DType::BF16
);

#[cfg(feature = "cubecl")]
make_element!(
    ty flex32,
    convert |elem: &dyn ToElement| flex32::from_f32(elem.to_f32()),
    random |distribution: Distribution, rng: &mut R| {
        let sample: f32 = distribution.sampler(rng).sample();
        flex32::from_elem(sample)
    },
    cmp |a: &flex32, b: &flex32| a.total_cmp(b),
    dtype DType::Flex32,
    min flex32::from_f32(f16::MIN.to_f32_const()),
    max flex32::from_f32(f16::MAX.to_f32_const())
);

make_element!(
    ty bool,
    convert ToElement::to_bool,
    random |distribution: Distribution, rng: &mut R| {
        let sample: u8 = distribution.sampler(rng).sample();
        bool::from_elem(sample)
    },
    cmp |a: &bool, b: &bool| Ord::cmp(a, b),
    dtype DType::Bool,
    min false,
    max true
);
