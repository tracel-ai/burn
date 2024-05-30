use super::{BinaryOperator, UnaryOperator, Variable};
use serde::{Deserialize, Serialize};

/// All subcube operations.
///
/// Note that not all backends support wrap/subgroup operations. Use the [runtime flag](crate::Runtime::subcube).
///
/// See [the wgsl subcube proposal](https://github.com/gpuweb/gpuweb/blob/a96d0c18b96fa3328e688ea54e17e9f4df24abe1/proposals/subcubes.md)
/// for reference.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(dead_code, missing_docs)] // Some variants might not be used with different flags
pub enum Subcube {
    SubcubeElect(SubgroupNoInput),
    SubcubeAll(UnaryOperator),
    SubcubeAny(UnaryOperator),
    SubcubeBroadcast(BinaryOperator),
    SubcubeSum(UnaryOperator),
    SubcubeProduct(UnaryOperator),
    SubcubeAnd(UnaryOperator),
    SubcubeOr(UnaryOperator),
    SubcubeXor(UnaryOperator),
    SubcubeMin(UnaryOperator),
    SubcubeMax(UnaryOperator),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct SubgroupNoInput {
    pub out: Variable,
}
