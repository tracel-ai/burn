use std::fmt::Display;

use super::Body;
use crate::kernel::WORKGROUP_DEFAULT;
use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub enum Location {
    Storage,
    #[allow(dead_code)]
    Workgroup,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub enum Visibility {
    Read,
    ReadWrite,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy, Serialize, Deserialize)]
pub enum Elem {
    F32,
    I32,
    U32,
    Bool,
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

#[derive(Debug, Clone, PartialEq, Eq, Copy, Serialize, Deserialize)]
pub enum Item {
    Vec4(Elem),
    Vec3(Elem),
    Vec2(Elem),
    Scalar(Elem),
}

impl Item {
    pub fn elem(&self) -> Elem {
        match self {
            Self::Vec4(elem) => *elem,
            Self::Vec3(elem) => *elem,
            Self::Vec2(elem) => *elem,
            Self::Scalar(elem) => *elem,
        }
    }
}

impl Elem {
    /// Returns the size of the elem type in bytes.
    pub fn size(&self) -> usize {
        match self {
            Elem::F32 => core::mem::size_of::<f32>(),
            Elem::I32 => core::mem::size_of::<i32>(),
            Elem::U32 => core::mem::size_of::<u32>(),
            Elem::Bool => core::mem::size_of::<bool>(),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
pub struct Binding {
    pub location: Location,
    pub visibility: Visibility,
    pub item: Item,
    pub size: Option<usize>,
}

#[derive(new, Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
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
pub struct ComputeShader {
    pub inputs: Vec<Binding>,
    pub outputs: Vec<Binding>,
    pub named: Vec<(String, Binding)>,
    pub workgroup_size: WorkgroupSize,
    pub global_invocation_id: bool,
    pub num_workgroups: bool,
    pub body: Body,
}
