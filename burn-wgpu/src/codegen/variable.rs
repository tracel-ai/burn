use super::{Item, Vectorization};
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
    pub fn vectorize(&self, vectorize: Vectorization) -> Self {
        match vectorize {
            Vectorization::Vec4 => match self {
                Variable::Input(id, ty) => Variable::Input(*id, Item::Vec4(ty.elem())),
                Variable::Local(id, ty) => Variable::Local(*id, Item::Vec4(ty.elem())),
                Variable::Output(id, ty) => Variable::Output(*id, Item::Vec4(ty.elem())),
                Variable::Constant(id, ty) => Variable::Constant(*id, Item::Vec4(ty.elem())),
                Variable::Scalar(_, _) => self.clone(),
            },
            Vectorization::Vec3 => match self {
                Variable::Input(id, ty) => Variable::Input(*id, Item::Vec3(ty.elem())),
                Variable::Local(id, ty) => Variable::Local(*id, Item::Vec3(ty.elem())),
                Variable::Output(id, ty) => Variable::Output(*id, Item::Vec3(ty.elem())),
                Variable::Constant(id, ty) => Variable::Constant(*id, Item::Vec3(ty.elem())),
                Variable::Scalar(_, _) => self.clone(),
            },
            Vectorization::Vec2 => match self {
                Variable::Input(id, ty) => Variable::Input(*id, Item::Vec2(ty.elem())),
                Variable::Local(id, ty) => Variable::Local(*id, Item::Vec2(ty.elem())),
                Variable::Output(id, ty) => Variable::Output(*id, Item::Vec2(ty.elem())),
                Variable::Constant(id, ty) => Variable::Constant(*id, Item::Vec2(ty.elem())),
                Variable::Scalar(_, _) => self.clone(),
            },
            Vectorization::Scalar => match self {
                Variable::Input(id, ty) => Variable::Input(*id, Item::Scalar(ty.elem())),
                Variable::Local(id, ty) => Variable::Local(*id, Item::Scalar(ty.elem())),
                Variable::Output(id, ty) => Variable::Output(*id, Item::Scalar(ty.elem())),
                Variable::Constant(id, ty) => Variable::Constant(*id, Item::Scalar(ty.elem())),
                Variable::Scalar(_, _) => self.clone(),
            },
        }
    }

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
