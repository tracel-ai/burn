use crate::{tensor::ops::*, Distribution};
use half::f16;
use num_traits::ToPrimitive;
use rand::prelude::StdRng;

pub trait Element:
    Zeros
    + ToPrimitive
    + ElementRandom
    + ElementConversion
    + ElementPrecision
    + ElementValue
    + Ones
    + std::ops::Mul<Self, Output = Self>
    + std::fmt::Debug
    + Default
    + Send
    + Sync
    + Copy
    + std::cmp::PartialOrd<Self>
    + 'static
{
}

pub trait ElementConversion {
    fn from_elem<E: ToPrimitive>(elem: E) -> Self;
    fn to_elem<E: Element>(&self) -> E;
}

pub trait ElementRandom {
    fn random(distribution: Distribution<Self>, rng: &mut StdRng) -> Self
    where
        Self: Sized;
}

pub trait ElementValue {
    fn inf() -> Self;
    fn inf_neg() -> Self;
    fn nan() -> Self;
    fn zero() -> Self;
    fn one() -> Self;
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
        zero $zero:expr,
        one $one:expr,
        convert $convert:expr,
        random $random:expr

    ) => {
        impl Element for $type {}

        impl Zeros for $type {
            fn zeros(&self) -> $type {
                $zero
            }
        }

        impl Ones for $type {
            fn ones(&self) -> $type {
                $one
            }
        }

        impl ElementConversion for $type {
            fn from_elem<E: ToPrimitive>(elem: E) -> Self {
                $convert(&elem)
            }
            fn to_elem<E: Element>(&self) -> E {
                E::from_elem(*self)
            }
        }

        impl ElementValue for $type {
            fn inf() -> Self {
                Self::from_elem(f64::INFINITY)
            }
            fn inf_neg() -> Self {
                Self::from_elem(std::ops::Neg::neg(f64::INFINITY))
            }
            fn nan() -> Self {
                Self::from_elem(f64::NAN)
            }
            fn zero() -> Self {
                $zero
            }
            fn one() -> Self {
                $one
            }
        }

        impl ElementPrecision for $type {
            fn precision() -> Precision {
                $precision
            }
        }

        impl ElementRandom for $type {
            fn random(distribution: Distribution<Self>, rng: &mut StdRng) -> Self {
                $random(distribution, rng)
            }
        }

            };
    (
        float $float:ident $precision:expr,
        convert $convert:expr,
        random $random:expr
    ) => {
        make_element!(
            ty $float $precision,
            zero 0.0,
            one 1.0,
            convert $convert,
            random $random
        );
    };
    (
        int $int:ident $precision:expr,
        convert $convert:expr,
        random $random:expr
    ) => {
        make_element!(
            ty $int $precision,
            zero 0,
            one 1,
            convert $convert,
            random $random
        );
    };
}

make_element!(
    float f64 Precision::Double,
    convert |elem: &dyn ToPrimitive| elem.to_f64().unwrap(),
    random |distribution: Distribution<f64>, rng: &mut StdRng| distribution.sampler(rng).sample()
);

make_element!(
    float f32 Precision::Full,
    convert |elem: &dyn ToPrimitive| elem.to_f32().unwrap(),
    random |distribution: Distribution<f32>, rng: &mut StdRng| distribution.sampler(rng).sample()
);

make_element!(
    int i64 Precision::Double,
    convert |elem: &dyn ToPrimitive| elem.to_i64().unwrap(),
    random |distribution: Distribution<i64>, rng: &mut StdRng| distribution.sampler(rng).sample()
);
make_element!(
    int i32 Precision::Full,
    convert |elem: &dyn ToPrimitive| elem.to_i32().unwrap(),
    random |distribution: Distribution<i32>, rng: &mut StdRng| distribution.sampler(rng).sample()
);
make_element!(
    int i16 Precision::Half,
    convert |elem: &dyn ToPrimitive| elem.to_i16().unwrap(),
    random |distribution: Distribution<i16>, rng: &mut StdRng| distribution.sampler(rng).sample()
);
make_element!(
    int i8 Precision::Other,
    convert |elem: &dyn ToPrimitive| elem.to_i8().unwrap(),
    random |distribution: Distribution<i8>, rng: &mut StdRng| distribution.sampler(rng).sample()
);

make_element!(
    int u8 Precision::Other,
    convert |elem: &dyn ToPrimitive| elem.to_u8().unwrap(),
    random |distribution: Distribution<u8>, rng: &mut StdRng| distribution.sampler(rng).sample()
);
make_element!(
    ty f16 Precision::Half,
    zero <f16 as num_traits::Zero>::zero(),
    one <f16 as num_traits::One>::one(),
    convert |elem: &dyn ToPrimitive| f16::from_f32(elem.to_f32().unwrap()),
    random |distribution: Distribution<f16>, rng: &mut StdRng| {
        let distribution: Distribution<f32> = distribution.convert();
        let sample = distribution.sampler(rng).sample();
        f16::from_elem(sample)
    }
);
