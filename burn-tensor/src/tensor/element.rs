use crate::Distribution;
use half::{bf16, f16};
use num_traits::ToPrimitive;
use rand::RngCore;

pub trait Element:
    ToPrimitive
    + ElementRandom
    + ElementConversion
    + ElementPrecision
    + core::fmt::Debug
    + core::fmt::Display
    + Default
    + Send
    + Sync
    + Copy
    + 'static
{
}

pub trait ElementConversion {
    fn from_elem<E: ToPrimitive>(elem: E) -> Self;
    fn elem<E: Element>(self) -> E;
}

pub trait ElementRandom {
    fn random<R: RngCore>(distribution: Distribution<Self>, rng: &mut R) -> Self
    where
        Self: Sized;
}

#[derive(Clone, PartialEq, Eq, Copy, Debug)]
pub enum Precision {
    Double,
    Full,
    Half,
    Other,
}

pub trait ElementPrecision {
    fn precision() -> Precision;
}

#[macro_export]
macro_rules! make_element {
    (
        ty $type:ident $precision:expr,
        convert $convert:expr,
        random $random:expr

    ) => {
        impl Element for $type {}

        impl ElementConversion for $type {
            fn from_elem<E: ToPrimitive>(elem: E) -> Self {
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
            fn random<R: RngCore>(distribution: Distribution<Self>, rng: &mut R) -> Self {
                $random(distribution, rng)
            }
        }
    };
}

make_element!(
    ty f64 Precision::Double,
    convert |elem: &dyn ToPrimitive| elem.to_f64().unwrap(),
    random |distribution: Distribution<f64>, rng: &mut R| distribution.sampler(rng).sample()
);

make_element!(
    ty f32 Precision::Full,
    convert |elem: &dyn ToPrimitive| elem.to_f32().unwrap(),
    random |distribution: Distribution<f32>, rng: &mut R| distribution.sampler(rng).sample()
);

make_element!(
    ty i64 Precision::Double,
    convert |elem: &dyn ToPrimitive| elem.to_i64().unwrap(),
    random |distribution: Distribution<i64>, rng: &mut R| distribution.sampler(rng).sample()
);

make_element!(
    ty i32 Precision::Full,
    convert |elem: &dyn ToPrimitive| elem.to_i32().unwrap(),
    random |distribution: Distribution<i32>, rng: &mut R| distribution.sampler(rng).sample()
);

make_element!(
    ty i16 Precision::Half,
    convert |elem: &dyn ToPrimitive| elem.to_i16().unwrap(),
    random |distribution: Distribution<i16>, rng: &mut R| distribution.sampler(rng).sample()
);

make_element!(
    ty i8 Precision::Other,
    convert |elem: &dyn ToPrimitive| elem.to_i8().unwrap(),
    random |distribution: Distribution<i8>, rng: &mut R| distribution.sampler(rng).sample()
);

make_element!(
    ty u8 Precision::Other,
    convert |elem: &dyn ToPrimitive| elem.to_u8().unwrap(),
    random |distribution: Distribution<u8>, rng: &mut R| distribution.sampler(rng).sample()
);

make_element!(
    ty f16 Precision::Half,
    convert |elem: &dyn ToPrimitive| f16::from_f32(elem.to_f32().unwrap()),
    random |distribution: Distribution<f16>, rng: &mut R| {
        let distribution: Distribution<f32> = distribution.convert();
        let sample = distribution.sampler(rng).sample();
        f16::from_elem(sample)
    }
);
make_element!(
    ty bf16 Precision::Half,
    convert |elem: &dyn ToPrimitive| bf16::from_f32(elem.to_f32().unwrap()),
    random |distribution: Distribution<bf16>, rng: &mut R| {
        let distribution: Distribution<f32> = distribution.convert();
        let sample = distribution.sampler(rng).sample();
        bf16::from_elem(sample)
    }
);
