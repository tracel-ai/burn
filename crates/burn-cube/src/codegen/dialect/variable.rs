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
    pub fn index(&self) -> Option<u16> {
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

    /// Fetch the item of the variable.
    pub fn item(&self) -> Item {
        match self {
            Variable::GlobalInputArray(_, item) => *item,
            Variable::GlobalOutputArray(_, item) => *item,
            Variable::GlobalScalar(_, elem) => Item::scalar(*elem),
            Variable::Local(_, item, _) => *item,
            Variable::LocalScalar(_, elem, _) => Item::scalar(*elem),
            Variable::ConstantScalar(_, elem) => Item::scalar(*elem),
            Variable::SharedMemory(_, item, _) => *item,
            Variable::LocalArray(_, item, _, _) => *item,
            Variable::Id => Item::scalar(Elem::UInt),
            Variable::Rank => Item::scalar(Elem::UInt),
            Variable::LocalInvocationIndex => Item::scalar(Elem::UInt),
            Variable::LocalInvocationIdX => Item::scalar(Elem::UInt),
            Variable::LocalInvocationIdY => Item::scalar(Elem::UInt),
            Variable::LocalInvocationIdZ => Item::scalar(Elem::UInt),
            Variable::WorkgroupIdX => Item::scalar(Elem::UInt),
            Variable::WorkgroupIdY => Item::scalar(Elem::UInt),
            Variable::WorkgroupIdZ => Item::scalar(Elem::UInt),
            Variable::GlobalInvocationIdX => Item::scalar(Elem::UInt),
            Variable::GlobalInvocationIdY => Item::scalar(Elem::UInt),
            Variable::GlobalInvocationIdZ => Item::scalar(Elem::UInt),
            Variable::WorkgroupSizeX => Item::scalar(Elem::UInt),
            Variable::WorkgroupSizeY => Item::scalar(Elem::UInt),
            Variable::WorkgroupSizeZ => Item::scalar(Elem::UInt),
            Variable::NumWorkgroupsX => Item::scalar(Elem::UInt),
            Variable::NumWorkgroupsY => Item::scalar(Elem::UInt),
            Variable::NumWorkgroupsZ => Item::scalar(Elem::UInt),
        }
    }
}

// Useful with the cube_inline macro.
impl From<&Variable> for Variable {
    fn from(value: &Variable) -> Self {
        *value
    }
}
