use crate::{ExpandElement, RuntimeType};

pub trait Float:
    Clone
    + Copy
    + RuntimeType<ExpandType = ExpandElement>
    + std::cmp::PartialOrd
    + std::ops::Add<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Sub<Output = Self>
{
    fn into_kind() -> burn_jit::gpu::FloatKind;
    fn new(val: f32, vectorization: usize) -> Self;
}

#[derive(Clone, Copy)]
pub struct F16 {
    pub val: f32,
    pub vectorization: usize,
}
#[derive(Clone, Copy)]
pub struct BF16 {
    pub val: f32,
    pub vectorization: usize,
}
#[derive(Clone, Copy)]
pub struct F32 {
    pub val: f32,
    pub vectorization: usize,
}
#[derive(Clone, Copy)]
pub struct F64 {
    pub val: f32,
    pub vectorization: usize,
}

impl RuntimeType for F16 {
    type ExpandType = ExpandElement;
}

impl RuntimeType for BF16 {
    type ExpandType = ExpandElement;
}

impl RuntimeType for F32 {
    type ExpandType = ExpandElement;
}

impl RuntimeType for F64 {
    type ExpandType = ExpandElement;
}

impl Float for F16 {
    fn into_kind() -> burn_jit::gpu::FloatKind {
        burn_jit::gpu::FloatKind::F16
    }
    fn new(val: f32, vectorization: usize) -> Self {
        Self { val, vectorization }
    }
}
impl Float for BF16 {
    fn into_kind() -> burn_jit::gpu::FloatKind {
        burn_jit::gpu::FloatKind::BF16
    }
    fn new(val: f32, vectorization: usize) -> Self {
        Self { val, vectorization }
    }
}
impl Float for F32 {
    fn into_kind() -> burn_jit::gpu::FloatKind {
        burn_jit::gpu::FloatKind::F32
    }
    fn new(val: f32, vectorization: usize) -> Self {
        Self { val, vectorization }
    }
}
impl Float for F64 {
    fn into_kind() -> burn_jit::gpu::FloatKind {
        burn_jit::gpu::FloatKind::F64
    }
    fn new(val: f32, vectorization: usize) -> Self {
        Self { val, vectorization }
    }
}
