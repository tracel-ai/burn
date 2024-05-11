use crate::{CubeContext, ExpandElement, RuntimeType};

pub trait Numeric:
    Clone
    + Copy
    + RuntimeType
    + std::ops::Add<Output = Self>
    + std::ops::AddAssign
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::cmp::PartialOrd
{
    // If we use numeric then constants are necessarily ints
    fn new(val: i64) -> Self;
    fn new_expand(context: &mut CubeContext, val: i64) -> ExpandElement;
}
