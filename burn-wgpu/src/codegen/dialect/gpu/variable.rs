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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexedVariable {
    var: Variable,
    index: usize,
}

impl Variable {
    pub fn index(&self, index: usize) -> IndexedVariable {
        IndexedVariable {
            var: self.clone(),
            index,
        }
    }

    pub fn item(&self) -> &Item {
        match self {
            Variable::Input(_, e) => e,
            Variable::Scalar(_, e) => e,
            Variable::Local(_, e) => e,
            Variable::Output(_, e) => e,
            Variable::Constant(_, e) => e,
        }
    }
}
