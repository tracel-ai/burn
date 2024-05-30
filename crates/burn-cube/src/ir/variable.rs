use super::{Elem, Item};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub enum Variable {
    Rank,
    GlobalInputArray(u16, Item),
    GlobalScalar(u16, Elem),
    GlobalOutputArray(u16, Item),
    Local(u16, Item, u8),
    LocalScalar(u16, Elem, u8),
    ConstantScalar(f64, Elem),
    SharedMemory(u16, Item, u32),
    LocalArray(u16, Item, u8, u32),
    UnitPos,
    UnitPosX,
    UnitPosY,
    UnitPosZ,
    CubePos,
    CubePosX,
    CubePosY,
    CubePosZ,
    CubeDim,
    CubeDimX,
    CubeDimY,
    CubeDimZ,
    CubeCount,
    CubeCountX,
    CubeCountY,
    CubeCountZ,
    SubcubeDim,
    AbsolutePos,
    AbsolutePosX,
    AbsolutePosY,
    AbsolutePosZ,
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
            Variable::AbsolutePos => None,
            Variable::UnitPos => None,
            Variable::UnitPosX => None,
            Variable::UnitPosY => None,
            Variable::UnitPosZ => None,
            Variable::Rank => None,
            Variable::CubePosX => None,
            Variable::CubePosY => None,
            Variable::CubePosZ => None,
            Variable::AbsolutePosX => None,
            Variable::AbsolutePosY => None,
            Variable::AbsolutePosZ => None,
            Variable::CubeDimX => None,
            Variable::CubeDimY => None,
            Variable::CubeDimZ => None,
            Variable::CubeCountX => None,
            Variable::CubeCountY => None,
            Variable::CubeCountZ => None,
            Variable::CubePos => None,
            Variable::CubeCount => None,
            Variable::CubeDim => None,
            Variable::SubcubeDim => None,
        }
    }

    /// Fetch the item of the variable.
    pub fn item(&self) -> Item {
        match self {
            Variable::GlobalInputArray(_, item) => *item,
            Variable::GlobalOutputArray(_, item) => *item,
            Variable::GlobalScalar(_, elem) => Item::new(*elem),
            Variable::Local(_, item, _) => *item,
            Variable::LocalScalar(_, elem, _) => Item::new(*elem),
            Variable::ConstantScalar(_, elem) => Item::new(*elem),
            Variable::SharedMemory(_, item, _) => *item,
            Variable::LocalArray(_, item, _, _) => *item,
            Variable::AbsolutePos => Item::new(Elem::UInt),
            Variable::Rank => Item::new(Elem::UInt),
            Variable::UnitPos => Item::new(Elem::UInt),
            Variable::UnitPosX => Item::new(Elem::UInt),
            Variable::UnitPosY => Item::new(Elem::UInt),
            Variable::UnitPosZ => Item::new(Elem::UInt),
            Variable::CubePosX => Item::new(Elem::UInt),
            Variable::CubePosY => Item::new(Elem::UInt),
            Variable::CubePosZ => Item::new(Elem::UInt),
            Variable::AbsolutePosX => Item::new(Elem::UInt),
            Variable::AbsolutePosY => Item::new(Elem::UInt),
            Variable::AbsolutePosZ => Item::new(Elem::UInt),
            Variable::CubeDimX => Item::new(Elem::UInt),
            Variable::CubeDimY => Item::new(Elem::UInt),
            Variable::CubeDimZ => Item::new(Elem::UInt),
            Variable::CubeCountX => Item::new(Elem::UInt),
            Variable::CubeCountY => Item::new(Elem::UInt),
            Variable::CubeCountZ => Item::new(Elem::UInt),
            Variable::CubePos => Item::new(Elem::UInt),
            Variable::CubeCount => Item::new(Elem::UInt),
            Variable::CubeDim => Item::new(Elem::UInt),
            Variable::SubcubeDim => Item::new(Elem::UInt),
        }
    }
}

// Useful with the cube_inline macro.
impl From<&Variable> for Variable {
    fn from(value: &Variable) -> Self {
        *value
    }
}
