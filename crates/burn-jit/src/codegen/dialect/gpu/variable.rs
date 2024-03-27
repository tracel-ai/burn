use super::{Elem, Item};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub enum Variable {
    GlobalInputArray(u16, Item),
    GlobalScalar(u16, Elem),
    GlobalOutputArray(u16, Item),
    Local(u16, Item, u8),
    LocalScalar(u16, Elem, u8),
    ConstantScalar(f64, Elem),
    SharedMemory(u16, Item, u32),
    LocalArray(u16, Item, u8, u32),
    Id,
    LocalInvocationIndex,
    LocalInvocationIdX,
    LocalInvocationIdY,
    LocalInvocationIdZ,
    WorkgroupIdX,
    WorkgroupIdY,
    WorkgroupIdZ,
    GlobalInvocationIdX,
    GlobalInvocationIdY,
    GlobalInvocationIdZ,
    Rank,
    WorkgroupSizeX,
    WorkgroupSizeY,
    WorkgroupSizeZ,
    NumWorkgroupsX,
    NumWorkgroupsY,
    NumWorkgroupsZ,
}

impl Variable {
    pub(crate) fn index(&self) -> Option<u16> {
        match self {
            Variable::GlobalInputArray(idx, _) => Some(*idx),
            Variable::GlobalScalar(idx, _) => Some(*idx),
            Variable::Local(idx, _, _) => Some(*idx),
            Variable::LocalScalar(idx, _, _) => Some(*idx),
            Variable::GlobalOutputArray(idx, _) => Some(*idx),
            Variable::ConstantScalar(_, _) => None,
            Variable::SharedMemory(idx, _, _) => Some(*idx),
            Variable::LocalArray(idx, _, _, _) => Some(*idx),
            Variable::Id => None,
            Variable::LocalInvocationIndex => None,
            Variable::LocalInvocationIdX => None,
            Variable::LocalInvocationIdY => None,
            Variable::LocalInvocationIdZ => None,
            Variable::Rank => None,
            Variable::WorkgroupIdX => None,
            Variable::WorkgroupIdY => None,
            Variable::WorkgroupIdZ => None,
            Variable::GlobalInvocationIdX => None,
            Variable::GlobalInvocationIdY => None,
            Variable::GlobalInvocationIdZ => None,
            Variable::WorkgroupSizeX => None,
            Variable::WorkgroupSizeY => None,
            Variable::WorkgroupSizeZ => None,
            Variable::NumWorkgroupsX => None,
            Variable::NumWorkgroupsY => None,
            Variable::NumWorkgroupsZ => None,
        }
    }
    pub(crate) fn item(&self) -> Item {
        match self {
            Variable::GlobalInputArray(_, item) => *item,
            Variable::GlobalOutputArray(_, item) => *item,
            Variable::GlobalScalar(_, elem) => Item::Scalar(*elem),
            Variable::Local(_, item, _) => *item,
            Variable::LocalScalar(_, elem, _) => Item::Scalar(*elem),
            Variable::ConstantScalar(_, elem) => Item::Scalar(*elem),
            Variable::SharedMemory(_, item, _) => *item,
            Variable::LocalArray(_, item, _, _) => *item,
            Variable::Id => Item::Scalar(Elem::UInt),
            Variable::Rank => Item::Scalar(Elem::UInt),
            Variable::LocalInvocationIndex => Item::Scalar(Elem::UInt),
            Variable::LocalInvocationIdX => Item::Scalar(Elem::UInt),
            Variable::LocalInvocationIdY => Item::Scalar(Elem::UInt),
            Variable::LocalInvocationIdZ => Item::Scalar(Elem::UInt),
            Variable::WorkgroupIdX => Item::Scalar(Elem::UInt),
            Variable::WorkgroupIdY => Item::Scalar(Elem::UInt),
            Variable::WorkgroupIdZ => Item::Scalar(Elem::UInt),
            Variable::GlobalInvocationIdX => Item::Scalar(Elem::UInt),
            Variable::GlobalInvocationIdY => Item::Scalar(Elem::UInt),
            Variable::GlobalInvocationIdZ => Item::Scalar(Elem::UInt),
            Variable::WorkgroupSizeX => Item::Scalar(Elem::UInt),
            Variable::WorkgroupSizeY => Item::Scalar(Elem::UInt),
            Variable::WorkgroupSizeZ => Item::Scalar(Elem::UInt),
            Variable::NumWorkgroupsX => Item::Scalar(Elem::UInt),
            Variable::NumWorkgroupsY => Item::Scalar(Elem::UInt),
            Variable::NumWorkgroupsZ => Item::Scalar(Elem::UInt),
        }
    }
}

// Useful with the gpu! macro.
impl From<&Variable> for Variable {
    fn from(value: &Variable) -> Self {
        *value
    }
}
