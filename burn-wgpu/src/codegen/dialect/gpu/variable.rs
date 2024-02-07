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
