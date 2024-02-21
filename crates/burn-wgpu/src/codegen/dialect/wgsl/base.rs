use crate::codegen::dialect::gpu;
use std::fmt::Display;

#[derive(Debug, Clone)]
pub enum Variable {
    GlobalInputArray(u16, Item),
    GlobalOutputArray(u16, Item),
    GlobalScalar(u16, Elem, gpu::Elem),
    ConstantScalar(f64, Elem),
    Local {
        index: u16,
        item: Item,
        scope_depth: u8,
    },
    LocalScalar {
        index: u16,
        elem: Elem,
        scope_depth: u8,
    },
    Id,
    Rank,
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
    pub fn is_always_scalar(&self) -> bool {
        match self {
            Variable::GlobalScalar(_, _, _) => true,
            Variable::ConstantScalar(_, _) => true,
            Variable::LocalScalar {
                index: _,
                elem: _,
                scope_depth: _,
            } => true,
            Variable::Id => true,
            Variable::Rank => true,
            Variable::GlobalInputArray(_, _) => false,
            Variable::GlobalOutputArray(_, _) => false,
            Variable::Local {
                index: _,
                item: _,
                scope_depth: _,
            } => false,
        }
    }
    pub fn index(&self, index: usize) -> IndexedVariable {
        IndexedVariable {
            var: self.clone(),
            index,
        }
    }

    pub fn item(&self) -> Item {
        match self {
            Self::GlobalInputArray(_, e) => *e,
            Self::GlobalOutputArray(_, e) => *e,
            Self::Local {
                index: _,
                item,
                scope_depth: _,
            } => *item,
            Self::ConstantScalar(_, e) => Item::Scalar(*e),
            Self::GlobalScalar(_, e, _) => Item::Scalar(*e),
            Self::Id => Item::Scalar(Elem::U32),
            Self::Rank => Item::Scalar(Elem::U32),
            Self::LocalScalar {
                index: _,
                elem,
                scope_depth: _,
            } => Item::Scalar(*elem),
        }
    }
    pub fn elem(&self) -> Elem {
        *self.item().elem()
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
            Variable::GlobalInputArray(number, _) => f.write_fmt(format_args!("input_{number}")),
            Variable::LocalScalar {
                index,
                elem: _,
                scope_depth,
            } => f.write_fmt(format_args!("s_{scope_depth}_{index}")),
            Variable::Local {
                index,
                item: _,
                scope_depth,
            } => f.write_fmt(format_args!("l_{scope_depth}_{index}")),
            Variable::GlobalOutputArray(number, _) => f.write_fmt(format_args!("output_{number}")),
            Variable::GlobalScalar(number, _, elem) => {
                f.write_fmt(format_args!("scalars_{elem}[{number}]"))
            }
            Variable::ConstantScalar(number, elem) => f.write_fmt(format_args!("{elem}({number})")),
            Variable::Id => f.write_str("id"),
            Variable::Rank => f.write_str("rank"),
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
            Variable::GlobalScalar(_, _, _) => f.write_fmt(format_args!("{var}")),
            _ => match should_index(&item) {
                true => f.write_fmt(format_args!("{var}[{index}]")),
                false => f.write_fmt(format_args!("{var}")),
            },
        }
    }
}
