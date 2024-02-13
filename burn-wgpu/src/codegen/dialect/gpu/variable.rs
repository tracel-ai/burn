use super::{Elem, Item};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Variable {
    Input(u16, Item),
    Scalar(u16, Item),
    Local(u16, Item, u8),
    Output(u16, Item),
    Constant(f64, Item),
    Id,
    Rank,
}

impl Variable {
    pub fn index(&self) -> Option<u16> {
        match self {
            Variable::Input(idx, _) => Some(*idx),
            Variable::Scalar(idx, _) => Some(*idx),
            Variable::Local(idx, _, _) => Some(*idx),
            Variable::Output(idx, _) => Some(*idx),
            Variable::Constant(_, _) => None,
            Variable::Id => None,
            Variable::Rank => None,
        }
    }
    pub fn item(&self) -> Item {
        match self {
            Variable::Input(_, item) => *item,
            Variable::Scalar(_, item) => *item,
            Variable::Local(_, item, _) => *item,
            Variable::Output(_, item) => *item,
            Variable::Constant(_, item) => *item,
            Variable::Id => Item::Scalar(Elem::UInt),
            Variable::Rank => Item::Scalar(Elem::UInt),
        }
    }
}
