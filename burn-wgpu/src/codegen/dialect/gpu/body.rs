use super::Operation;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, new)]
pub struct Body {
    pub operators: Vec<Operation>,
}
