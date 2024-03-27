use core::cmp::Ordering;

use crate::Distribution;
use half::{bf16, f16};
use num_traits::{identities::Zero, One, ToPrimitive};
use rand::RngCore;

/// Element trait for tensor.
pub trait Element:
    ToPrimitive
    + Zero
    + One
    + ElementRandom
    + ElementConversion
    + ElementPrecision
    + ElementComparison
    + core::fmt::Debug
    + core::fmt::Display
    + Default
    + Send
    + Sync
    + Copy
    + 'static
{
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
    fn from_elem<E: ToPrimitive>(elem: E) -> Self;

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
    fn random<R: RngCore>(distribution: Distribution, rng: &mut R) -> Self
    where
        Self: Sized;
}

/// Element ordering trait.
pub trait ElementComparison {
    /// Returns and [Ordering] between `self` and `other`.
    fn cmp(&self, other: &Self) -> Ordering;
}

/// Element precision trait for tensor.
#[derive(Clone, PartialEq, Eq, Copy, Debug)]
pub enum Precision {
    /// Double precision, e.g. f64.
    Double,

    /// Full precision, e.g. f32.
    Full,

    /// Half precision, e.g. f16.
    Half,

    /// Other precision.
    Other,
}

/// Element precision trait for tensor.
pub trait ElementPrecision {
    /// Returns the precision of the element.
    fn precision() -> Precision;
}

/// Macro to implement the element trait for a type.
#[macro_export]
macro_rules! make_element {
    (
        ty $type:ident $precision:expr,
        convert $convert:expr,
        random $random:expr,
        cmp $cmp:expr

    ) => {
        impl Element for $type {}

        impl ElementConversion for $type {
            fn from_elem<E: ToPrimitive>(elem: E) -> Self {
                #[allow(clippy::redundant_closure_call)]
                $convert(&elem)
            }
            fn elem<E: Element>(self) -> E {
                E::from_elem(self)
            }
        }

        impl ElementPrecision for $type {
            fn precision() -> Precision {
                $precision
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
    };
}

make_element!(
    ty f64 Precision::Double,
    convert |elem: &dyn ToPrimitive| elem.to_f64().unwrap(),
    random |distribution: Distribution, rng: &mut R| distribution.sampler(rng).sample(),
    cmp |a: &f64, b: &f64| a.total_cmp(b)
);

make_element!(
    ty f32 Precision::Full,
    convert |elem: &dyn ToPrimitive| elem.to_f32().unwrap(),
    random |distribution: Distribution, rng: &mut R| distribution.sampler(rng).sample(),
    cmp |a: &f32, b: &f32| a.total_cmp(b)
);

make_element!(
    ty i64 Precision::Double,
    convert |elem: &dyn ToPrimitive| elem.to_i64().unwrap(),
    random |distribution: Distribution, rng: &mut R| distribution.sampler(rng).sample(),
    cmp |a: &i64, b: &i64| Ord::cmp(a, b)
);

make_element!(
    ty i32 Precision::Full,
    convert |elem: &dyn ToPrimitive| elem.to_i32().unwrap(),
    random |distribution: Distribution, rng: &mut R| distribution.sampler(rng).sample(),
    cmp |a: &i32, b: &i32| Ord::cmp(a, b)
);

make_element!(
    ty u32 Precision::Full,
    convert |elem: &dyn ToPrimitive| elem.to_u32().unwrap(),
    random |distribution: Distribution, rng: &mut R| distribution.sampler(rng).sample(),
    cmp |a: &u32, b: &u32| Ord::cmp(a, b)
);

make_element!(
    ty i16 Precision::Half,
    convert |elem: &dyn ToPrimitive| elem.to_i16().unwrap(),
    random |distribution: Distribution, rng: &mut R| distribution.sampler(rng).sample(),
    cmp |a: &i16, b: &i16| Ord::cmp(a, b)
);

make_element!(
    ty i8 Precision::Other,
    convert |elem: &dyn ToPrimitive| elem.to_i8().unwrap(),
    random |distribution: Distribution, rng: &mut R| distribution.sampler(rng).sample(),
    cmp |a: &i8, b: &i8| Ord::cmp(a, b)
);

make_element!(
    ty u8 Precision::Other,
    convert |elem: &dyn ToPrimitive| elem.to_u8().unwrap(),
    random |distribution: Distribution, rng: &mut R| distribution.sampler(rng).sample(),
    cmp |a: &u8, b: &u8| Ord::cmp(a, b)
);

make_element!(
    ty f16 Precision::Half,
    convert |elem: &dyn ToPrimitive| f16::from_f32(elem.to_f32().unwrap()),
    random |distribution: Distribution, rng: &mut R| {
        let sample: f32 = distribution.sampler(rng).sample();
        f16::from_elem(sample)
    },
    cmp |a: &f16, b: &f16| a.total_cmp(b)
);
make_element!(
    ty bf16 Precision::Half,
    convert |elem: &dyn ToPrimitive| bf16::from_f32(elem.to_f32().unwrap()),
    random |distribution: Distribution, rng: &mut R| {
        let sample: f32 = distribution.sampler(rng).sample();
        bf16::from_elem(sample)
    },
    cmp |a: &bf16, b: &bf16| a.total_cmp(b)
);
