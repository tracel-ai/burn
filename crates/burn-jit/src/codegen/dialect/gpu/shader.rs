use super::Scope;
use crate::kernel::WORKGROUP_DEFAULT;
use burn_tensor::DType;
use serde::{Deserialize, Serialize};
use std::fmt::Display;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum Location {
    Storage,
    Workgroup,
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
        Item::Scalar(val)
    }
}

impl From<DType> for Elem {
    fn from(dtype: DType) -> Self {
        match dtype {
            DType::F64 => Elem::Float(FloatKind::F64),
            DType::F32 => Elem::Float(FloatKind::F32),
            DType::F16 => Elem::Float(FloatKind::F16),
            DType::BF16 => Elem::Float(FloatKind::BF16),
            DType::I64 => Elem::Int(IntKind::I64),
            DType::I32 => Elem::Int(IntKind::I32),
            DType::I16 => panic!("i16 isn't supported yet."),
            DType::I8 => panic!("i8 isn't supported yet."),
            DType::U64 => Elem::UInt,
            DType::U32 => Elem::UInt,
            DType::U8 => panic!("u8 isn't supported yet."),
            DType::Bool => Elem::Bool,
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
#[allow(missing_docs)]
pub enum Item {
    Vec4(Elem),
    Vec3(Elem),
    Vec2(Elem),
    Scalar(Elem),
}

impl Item {
    /// Fetch the elem of the item.
    pub fn elem(&self) -> Elem {
        match self {
            Self::Vec4(elem) => *elem,
            Self::Vec3(elem) => *elem,
            Self::Vec2(elem) => *elem,
            Self::Scalar(elem) => *elem,
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
pub struct WorkgroupSize {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Default for WorkgroupSize {
    fn default() -> Self {
        Self {
            x: WORKGROUP_DEFAULT as u32,
            y: WORKGROUP_DEFAULT as u32,
            z: 1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ComputeShader {
    pub inputs: Vec<Binding>,
    pub outputs: Vec<Binding>,
    pub named: Vec<(String, Binding)>,
    pub workgroup_size: WorkgroupSize,
    pub body: Scope,
}
