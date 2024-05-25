use burn_cube::dialect as gpu;
use half::{bf16, f16};
use std::fmt::Display;

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum Elem {
    F32,
    F16,
    BF16,
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

impl Display for Elem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Elem::F16 => f.write_str("f16"),
            Elem::F32 => f.write_str("float"),
            Elem::BF16 => f.write_str("bf16"),
            Elem::I32 => f.write_str("int"),
            Elem::U32 => f.write_str("uint"),
            Elem::Bool => f.write_str("bool"),
        }
    }
}

impl Display for Item {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Item::Vec4(elem) => match elem {
                Elem::F32 => f.write_str("float4"),
                Elem::I32 => f.write_str("int4"),
                Elem::U32 => f.write_str("uint4"),
                Elem::Bool => f.write_str("bool4"),
                Elem::BF16 => f.write_str("bf164"),
                Elem::F16 => f.write_str("f164"),
            },
            Item::Vec3(elem) => match elem {
                Elem::F32 => f.write_str("float3"),
                Elem::I32 => f.write_str("int3"),
                Elem::U32 => f.write_str("uint3"),
                Elem::Bool => f.write_str("bool3"),
                Elem::BF16 => f.write_str("bf163"),
                Elem::F16 => f.write_str("f163"),
            },
            Item::Vec2(elem) => match elem {
                Elem::F32 => f.write_str("float2"),
                Elem::I32 => f.write_str("int2"),
                Elem::U32 => f.write_str("uint2"),
                Elem::Bool => f.write_str("bool2"),
                Elem::BF16 => f.write_str("bf162"),
                Elem::F16 => f.write_str("f162"),
            },
            Item::Scalar(elem) => f.write_fmt(format_args!("{elem}")),
        }
    }
}

pub trait Component: Display {
    fn item(&self) -> Item;
    fn elem(&self) -> Elem {
        *self.item().elem()
    }
}

impl Component for IndexedVariable {
    fn item(&self) -> Item {
        self.var.item()
    }
}
impl Component for Variable {
    fn item(&self) -> Item {
        match self {
            Variable::GlobalInputArray(_, e) => *e,
            Variable::GlobalOutputArray(_, e) => *e,
            Variable::SharedMemory(_, e, _) => *e,
            Variable::Local {
                index: _,
                item,
                scope_depth: _,
            } => *item,
            Variable::ConstantScalar(_, e) => Item::Scalar(*e),
            Variable::GlobalScalar(_, e, _) => Item::Scalar(*e),
            Variable::Id => Item::Scalar(Elem::U32),
            Variable::LocalInvocationIndex => Item::Scalar(Elem::U32),
            Variable::LocalInvocationIdX => Item::Scalar(Elem::U32),
            Variable::LocalInvocationIdY => Item::Scalar(Elem::U32),
            Variable::LocalInvocationIdZ => Item::Scalar(Elem::U32),
            Variable::Rank => Item::Scalar(Elem::U32),
            Variable::LocalScalar {
                index: _,
                elem,
                scope_depth: _,
            } => Item::Scalar(*elem),
            Variable::WorkgroupIdX => Item::Scalar(Elem::U32),
            Variable::WorkgroupIdY => Item::Scalar(Elem::U32),
            Variable::WorkgroupIdZ => Item::Scalar(Elem::U32),
            Variable::GlobalInvocationIdX => Item::Scalar(Elem::U32),
            Variable::GlobalInvocationIdY => Item::Scalar(Elem::U32),
            Variable::GlobalInvocationIdZ => Item::Scalar(Elem::U32),
            Variable::WorkgroupSizeX => Item::Scalar(Elem::U32),
            Variable::WorkgroupSizeY => Item::Scalar(Elem::U32),
            Variable::WorkgroupSizeZ => Item::Scalar(Elem::U32),
            Variable::NumWorkgroupsX => Item::Scalar(Elem::U32),
            Variable::NumWorkgroupsY => Item::Scalar(Elem::U32),
            Variable::NumWorkgroupsZ => Item::Scalar(Elem::U32),
            Variable::LocalArray(_, e, _, _) => *e,
        }
    }
}

#[derive(Debug, Clone, Copy)]
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
    SharedMemory(u16, Item, u32),
    LocalArray(u16, Item, u8, u32),
    Id,
    LocalInvocationIndex,
    LocalInvocationIdX,
    LocalInvocationIdY,
    LocalInvocationIdZ,
    Rank,
    WorkgroupIdX,
    WorkgroupIdY,
    WorkgroupIdZ,
    GlobalInvocationIdX,
    GlobalInvocationIdY,
    GlobalInvocationIdZ,
    WorkgroupSizeX,
    WorkgroupSizeY,
    WorkgroupSizeZ,
    NumWorkgroupsX,
    NumWorkgroupsY,
    NumWorkgroupsZ,
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
            Variable::SharedMemory(number, _, _) => {
                f.write_fmt(format_args!("shared_memory_{number}"))
            }
            Variable::Id => f.write_str("id"),
            Variable::LocalInvocationIndex => f.write_str("invocationIndex"),
            Variable::LocalInvocationIdX => f.write_str("threadIdx.x"),
            Variable::LocalInvocationIdY => f.write_str("threadIdx.y"),
            Variable::LocalInvocationIdZ => f.write_str("threadIdx.z"),
            Variable::Rank => f.write_str("rank"),
            Variable::WorkgroupIdX => f.write_str("blockIdx.x"),
            Variable::WorkgroupIdY => f.write_str("blockIdx.y"),
            Variable::WorkgroupIdZ => f.write_str("blockIdx.z"),
            Variable::WorkgroupSizeX => f.write_str("blockDim.x"),
            Variable::WorkgroupSizeY => f.write_str("blockDim.y"),
            Variable::WorkgroupSizeZ => f.write_str("blockDim.z"),
            Variable::NumWorkgroupsX => f.write_str("gridDim.x"),
            Variable::NumWorkgroupsY => f.write_str("gridDim.y"),
            Variable::NumWorkgroupsZ => f.write_str("gridDim.z"),
            Variable::GlobalInvocationIdX => f.write_str("globalInvocationId.x"),
            Variable::GlobalInvocationIdY => f.write_str("globalInvocationId.y"),
            Variable::GlobalInvocationIdZ => f.write_str("globalInvocationId.z"),
            Variable::LocalArray(id, _item, depth, _size) => {
                f.write_fmt(format_args!("l_arr_{}_{}", id, depth))
            }
        }
    }
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
            Variable::LocalInvocationIndex => true,
            Variable::LocalInvocationIdX => true,
            Variable::LocalInvocationIdY => true,
            Variable::LocalInvocationIdZ => true,
            Variable::Rank => true,
            Variable::GlobalInputArray(_, _) => false,
            Variable::GlobalOutputArray(_, _) => false,
            Variable::SharedMemory(_, _, _) => false,
            Variable::Local {
                index: _,
                item: _,
                scope_depth: _,
            } => false,
            Variable::WorkgroupIdX => true,
            Variable::WorkgroupIdY => true,
            Variable::WorkgroupIdZ => true,
            Variable::GlobalInvocationIdX => true,
            Variable::GlobalInvocationIdY => true,
            Variable::GlobalInvocationIdZ => true,
            Variable::WorkgroupSizeX => true,
            Variable::WorkgroupSizeY => true,
            Variable::WorkgroupSizeZ => true,
            Variable::NumWorkgroupsX => true,
            Variable::NumWorkgroupsY => true,
            Variable::NumWorkgroupsZ => true,
            Variable::LocalArray(_, _, _, _) => false,
        }
    }

    pub fn index(&self, index: usize) -> IndexedVariable {
        IndexedVariable { var: *self, index }
    }
}

#[derive(Debug, Clone)]
pub struct IndexedVariable {
    var: Variable,
    index: usize,
}

impl Display for IndexedVariable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let var = &self.var;
        let item = self.var.item();

        match item {
            Item::Vec4(_) => match self.index {
                0 => f.write_fmt(format_args!("{var}.x"))?,
                1 => f.write_fmt(format_args!("{var}.y"))?,
                2 => f.write_fmt(format_args!("{var}.z"))?,
                3 => f.write_fmt(format_args!("{var}.w"))?,
                _ => unreachable!(),
            },
            Item::Vec3(_) => match self.index {
                0 => f.write_fmt(format_args!("{var}.x"))?,
                1 => f.write_fmt(format_args!("{var}.y"))?,
                2 => f.write_fmt(format_args!("{var}.z"))?,
                _ => unreachable!(),
            },
            Item::Vec2(_) => match self.index {
                0 => f.write_fmt(format_args!("{var}.x"))?,
                1 => f.write_fmt(format_args!("{var}.y"))?,
                _ => unreachable!(),
            },
            Item::Scalar(_) => f.write_fmt(format_args!("{var}"))?,
        }

        Ok(())
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
            Self::F16 => core::mem::size_of::<f16>(),
            Self::BF16 => core::mem::size_of::<bf16>(),
            Self::I32 => core::mem::size_of::<i32>(),
            Self::U32 => core::mem::size_of::<u32>(),
            Self::Bool => core::mem::size_of::<bool>(),
        }
    }
}
