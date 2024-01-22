use super::{Item, Vectorization};
use serde::{Deserialize, Serialize};
use std::fmt::Display;

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

impl Display for IndexedVariable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let should_index = |item: &Item| match item {
            Item::Vec4(_) => true,
            Item::Vec3(_) => true,
            Item::Vec2(_) => true,
            Item::Scalar(_) => false,
        };

        let var = &self.var;
        let item = var.item();
        let index = self.index;

        match self.var {
            Variable::Scalar(_, _) => f.write_fmt(format_args!("{var}")),
            _ => match should_index(item) {
                true => f.write_fmt(format_args!("{var}[{index}]")),
                false => f.write_fmt(format_args!("{var}")),
            },
        }
    }
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

impl Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Variable::Input(number, _) => f.write_fmt(format_args!("input_{number}")),
            Variable::Local(number, _) => f.write_fmt(format_args!("local_{number}")),
            Variable::Output(number, _) => f.write_fmt(format_args!("output_{number}")),
            Variable::Scalar(number, item) => f.write_fmt(format_args!("scalars_{item}[{number}]")),
            Variable::Constant(number, item) => match item {
                Item::Vec4(elem) => f.write_fmt(format_args!(
                    "
vec4(
    {elem}({number}),
    {elem}({number}),
    {elem}({number}),
    {elem}({number}),
)"
                )),
                Item::Vec3(elem) => f.write_fmt(format_args!(
                    "
vec3(
    {elem}({number}),
    {elem}({number}),
    {elem}({number}),
)"
                )),
                Item::Vec2(elem) => f.write_fmt(format_args!(
                    "
vec2(
    {elem}({number}),
    {elem}({number}),
)"
                )),
                Item::Scalar(elem) => f.write_fmt(format_args!("{elem}({number})")),
            },
        }
    }
}
