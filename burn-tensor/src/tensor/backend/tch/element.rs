use crate::ops::{Ones, Zeros};
use crate::{
    make_element, Distribution, Element, ElementConversion, ElementPrecision, ElementRandom,
    ElementValue, Precision,
};
use half::f16;
use num_traits::ToPrimitive;
use rand::rngs::StdRng;

pub(crate) trait TchElement: Element + tch::kind::Element {}

impl TchElement for f64 {}
impl TchElement for f32 {}
impl TchElement for f16 {}

impl TchElement for i64 {}
impl TchElement for i32 {}
impl TchElement for i16 {}

impl TchElement for u8 {}

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
