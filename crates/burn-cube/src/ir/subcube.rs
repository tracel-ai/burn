use super::{BinaryOperator, InitOperator, UnaryOperator};
use serde::{Deserialize, Serialize};

/// All subcube operations.
///
/// Note that not all backends support subcube (warp/subgroup) operations. Use the [runtime flag](crate::Feature::Subcube).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(dead_code, missing_docs)] // Some variants might not be used with different flags
pub enum Subcube {
    Elect(InitOperator),
    All(UnaryOperator),
    Any(UnaryOperator),
    Broadcast(BinaryOperator),
    Sum(UnaryOperator),
    Prod(UnaryOperator),
    And(UnaryOperator),
    Or(UnaryOperator),
    Xor(UnaryOperator),
    Min(UnaryOperator),
    Max(UnaryOperator),
}
