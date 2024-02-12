use super::Item;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Variable {
    Input(u16, Item),
    Scalar(u16, Item),
    Local(u16, Item),
    Output(u16, Item),
    Constant(f64, Item),
}

impl Variable {
    pub fn index(&self) -> Option<u16> {
        match self {
            Variable::Input(idx, _) => Some(*idx),
            Variable::Scalar(idx, _) => Some(*idx),
            Variable::Local(idx, _) => Some(*idx),
            Variable::Output(idx, _) => Some(*idx),
            Variable::Constant(_, _) => None,
        }
    }
}
