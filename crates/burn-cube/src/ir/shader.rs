use super::{Scope, Vectorization};
use crate::SUBCUBE_DIM_APPROX;
use serde::{Deserialize, Serialize};
use std::fmt::Display;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum Location {
    Storage,
    Cube,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum Visibility {
    Read,
    ReadWrite,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum FloatKind {
    F16,
    BF16,
    F32,
    F64,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum IntKind {
    I32,
    I64,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum Elem {
    Float(FloatKind),
    Int(IntKind),
    UInt,
    Bool,
}

impl From<Elem> for Item {
    fn from(val: Elem) -> Self {
        Item::new(val)
    }
}

#[cfg(feature = "tensor")]
impl From<burn_tensor::DType> for Elem {
    fn from(dtype: burn_tensor::DType) -> Self {
        match dtype {
            burn_tensor::DType::F64 => Elem::Float(FloatKind::F64),
            burn_tensor::DType::F32 => Elem::Float(FloatKind::F32),
            burn_tensor::DType::F16 => Elem::Float(FloatKind::F16),
            burn_tensor::DType::BF16 => Elem::Float(FloatKind::BF16),
            burn_tensor::DType::I64 => Elem::Int(IntKind::I64),
            burn_tensor::DType::I32 => Elem::Int(IntKind::I32),
            burn_tensor::DType::I16 => panic!("i16 isn't supported yet."),
            burn_tensor::DType::I8 => panic!("i8 isn't supported yet."),
            burn_tensor::DType::U64 => Elem::UInt,
            burn_tensor::DType::U32 => Elem::UInt,
            burn_tensor::DType::U8 => panic!("u8 isn't supported yet."),
            burn_tensor::DType::Bool => Elem::Bool,
        }
    }
}

impl Display for Elem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // NOTE: we'll eventually want to differentiate between int/float types
            Self::Float(_) => f.write_str("float"),
            Self::Int(_) => f.write_str("int"),
            Self::UInt => f.write_str("uint"),
            Self::Bool => f.write_str("bool"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Copy, Serialize, Deserialize, Hash)]
pub struct Item {
    pub elem: Elem,
    pub vectorization: Vectorization,
}

impl Item {
    /// Fetch the elem of the item.
    pub fn elem(&self) -> Elem {
        self.elem
    }

    /// Create a new item without vectorization
    pub fn new(elem: Elem) -> Self {
        Self {
            elem,
            vectorization: 1,
        }
    }

    /// Create a new item with vectorization
    pub fn vectorized(elem: Elem, vectorization: Vectorization) -> Self {
        Self {
            elem,
            vectorization,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct Binding {
    pub location: Location,
    pub visibility: Visibility,
    pub item: Item,
    pub size: Option<usize>,
}

#[derive(new, Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize, Hash)]
#[allow(missing_docs)]
pub struct CubeDim {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Default for CubeDim {
    fn default() -> Self {
        Self {
            x: SUBCUBE_DIM_APPROX as u32,
            y: SUBCUBE_DIM_APPROX as u32,
            z: 1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct KernelDefinition {
    pub inputs: Vec<Binding>,
    pub outputs: Vec<Binding>,
    pub named: Vec<(String, Binding)>,
    pub cube_dim: CubeDim,
    pub body: Scope,
}
