use std::collections::HashMap;

use candle_core::Tensor as CandleTensor;

/// A nested map/vector of tensors.
#[derive(Debug, Clone)]
pub enum NestedValue {
    /// The default value (typically for primitives like integers)
    Default,

    /// A string value
    String(String),

    /// A map of nested values (typically used for structs)
    Map(HashMap<String, NestedValue>),

    /// A tensor
    Tensor(CandleTensor),

    /// A vector of nested values (typically used for vector of structs)
    Vec(Vec<NestedValue>),
}

impl NestedValue {
    pub fn get_map(&self) -> Option<&HashMap<String, NestedValue>> {
        match self {
            NestedValue::Map(map) => Some(map),
            _ => None,
        }
    }

    pub fn get_vec(&self) -> Option<&Vec<NestedValue>> {
        match self {
            NestedValue::Vec(vec) => Some(vec),
            _ => None,
        }
    }

    fn get_tensor(&self) -> Option<&CandleTensor> {
        match self {
            NestedValue::Tensor(tensor) => Some(tensor),
            _ => None,
        }
    }

    fn get_tensor_mut(&mut self) -> Option<&mut CandleTensor> {
        match self {
            NestedValue::Tensor(tensor) => Some(tensor),
            _ => None,
        }
    }

    pub fn get_string(&self) -> Option<&String> {
        match self {
            NestedValue::String(string) => Some(string),
            _ => None,
        }
    }
}
