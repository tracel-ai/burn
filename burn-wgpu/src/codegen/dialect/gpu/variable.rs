use super::{Elem, Item};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Variable {
    GlobalInputArray(u16, Item),
    GlobalScalar(u16, Elem),
    GlobalOutputArray(u16, Item),
    Local(u16, Item, u8),
    LocalScalar(u16, Elem, u8),
    ConstantScalar(f64, Elem),
    Id,
    Rank,
}

impl Variable {
    pub fn const_value(&self) -> Option<f64> {
        match self {
            Variable::ConstantScalar(value, _) => Some(*value),
            _ => None,
        }
    }
    pub fn index(&self) -> Option<u16> {
        match self {
            Variable::GlobalInputArray(idx, _) => Some(*idx),
            Variable::GlobalScalar(idx, _) => Some(*idx),
            Variable::Local(idx, _, _) => Some(*idx),
            Variable::LocalScalar(idx, _, _) => Some(*idx),
            Variable::GlobalOutputArray(idx, _) => Some(*idx),
            Variable::ConstantScalar(_, _) => None,
            Variable::Id => None,
            Variable::Rank => None,
        }
    }
    pub fn item(&self) -> Item {
        match self {
            Variable::GlobalInputArray(_, item) => *item,
            Variable::GlobalOutputArray(_, item) => *item,
            Variable::GlobalScalar(_, elem) => Item::Scalar(*elem),
            Variable::Local(_, item, _) => *item,
            Variable::LocalScalar(_, elem, _) => Item::Scalar(*elem),
            Variable::ConstantScalar(_, elem) => Item::Scalar(*elem),
            Variable::Id => Item::Scalar(Elem::UInt),
            Variable::Rank => Item::Scalar(Elem::UInt),
        }
    }
}
