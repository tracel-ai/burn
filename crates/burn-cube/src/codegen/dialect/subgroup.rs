use super::{BinaryOperator, UnaryOperator, Variable};
use serde::{Deserialize, Serialize};

/// All subgroup operations.
///
/// Note that not all backends support subgroup operations. Use the [runtime flag](crate::Runtime::subgroup).
///
/// See [the wgsl subgroup proposal](https://github.com/gpuweb/gpuweb/blob/a96d0c18b96fa3328e688ea54e17e9f4df24abe1/proposals/subgroups.md)
/// for reference.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(dead_code, missing_docs)] // Some variants might not be used with different flags
pub enum Subgroup {
    SubgroupElect(SubgroupNoInput),
    SubgroupAll(UnaryOperator),
    SubgroupAny(UnaryOperator),
    SubgroupBroadcast(BinaryOperator),
    SubgroupSum(UnaryOperator),
    SubgroupProduct(UnaryOperator),
    SubgroupAnd(UnaryOperator),
    SubgroupOr(UnaryOperator),
    SubgroupXor(UnaryOperator),
    SubgroupMin(UnaryOperator),
    SubgroupMax(UnaryOperator),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct SubgroupNoInput {
    pub out: Variable,
}
