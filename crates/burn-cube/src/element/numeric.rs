// use crate::{BF16, F16, F32, F64, I32, I64};

use crate::{CubeContext, CubeType, ExpandElement};

pub trait Numeric:
    Clone + Copy + CubeType<ExpandType = ExpandElement> + std::ops::Add<Output = Self>
{
    fn new(val: f64) -> Self;
    fn new_expand(context: &mut CubeContext, val: f64) -> ExpandElement;
}
