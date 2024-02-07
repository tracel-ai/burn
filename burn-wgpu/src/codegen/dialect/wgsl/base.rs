use crate::codegen::dialect::gpu;
use std::fmt::Display;

#[derive(Debug, Clone)]
pub enum Variable {
    Input(u16, Item),
    Scalar(u16, Item, gpu::Elem),
    Local(u16, Item),
    Output(u16, Item),
    Constant(f64, Item),
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum Elem {
    F32,
    I32,
    U32,
    Bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum Item {
    Vec4(Elem),
    Vec3(Elem),
    Vec2(Elem),
    Scalar(Elem),
}

#[derive(Debug, Clone)]
pub struct IndexedVariable {
    var: Variable,
    index: usize,
}

impl Variable {
    pub fn index(&self, index: usize) -> IndexedVariable {
        IndexedVariable {
            var: self.clone(),
            index,
        }
    }

    pub fn item(&self) -> &Item {
        match self {
            Self::Input(_, e) => e,
            Self::Scalar(_, e, _) => e,
            Self::Local(_, e) => e,
            Self::Output(_, e) => e,
            Self::Constant(_, e) => e,
        }
    }
}

impl Item {
    pub fn elem(&self) -> &Elem {
        match self {
            Item::Vec4(e) => e,
            Item::Vec3(e) => e,
            Item::Vec2(e) => e,
            Item::Scalar(e) => e,
        }
    }
}

impl Elem {
    pub fn size(&self) -> usize {
        match self {
            Self::F32 => core::mem::size_of::<f32>(),
            Self::I32 => core::mem::size_of::<i32>(),
            Self::U32 => core::mem::size_of::<u32>(),
            Self::Bool => core::mem::size_of::<bool>(),
        }
    }
}

impl Display for Elem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F32 => f.write_str("f32"),
            Self::I32 => f.write_str("i32"),
            Self::U32 => f.write_str("u32"),
            Self::Bool => f.write_str("bool"),
        }
    }
}

impl Display for Item {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Item::Vec4(elem) => f.write_fmt(format_args!("vec4<{elem}>")),
            Item::Vec3(elem) => f.write_fmt(format_args!("vec3<{elem}>")),
            Item::Vec2(elem) => f.write_fmt(format_args!("vec2<{elem}>")),
            Item::Scalar(elem) => f.write_fmt(format_args!("{elem}")),
        }
    }
}

impl Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Variable::Input(number, _) => f.write_fmt(format_args!("input_{number}")),
            Variable::Local(number, _) => f.write_fmt(format_args!("local_{number}")),
            Variable::Output(number, _) => f.write_fmt(format_args!("output_{number}")),
            Variable::Scalar(number, _, elem) => {
                f.write_fmt(format_args!("scalars_{elem}[{number}]"))
            }
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
            Variable::Scalar(_, _, _) => f.write_fmt(format_args!("{var}")),
            _ => match should_index(item) {
                true => f.write_fmt(format_args!("{var}[{index}]")),
                false => f.write_fmt(format_args!("{var}")),
            },
        }
    }
}
