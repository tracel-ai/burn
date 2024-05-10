// use crate::{BF16, F16, F32, F64, I32, I64};

use burn_jit::gpu::Elem;

use crate::{CubeContext, CubeType, ExpandElement};

pub trait Numeric:
    Clone + Copy + CubeType<ExpandType = ExpandElement> + std::ops::Add<Output = Self>
{
    // If we use numeric then constants are necessarily ints
    fn new(val: i64) -> Self;
    fn new_expand(context: &mut CubeContext, val: i64) -> ExpandElement;

    fn into_elem() -> Elem;
}
