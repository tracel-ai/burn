use burn_jit::gpu::Variable;
use std::sync::Arc;

pub trait RuntimeType {
    type ExpandType: Clone;
}

pub type ExpandElement = Arc<Variable>;

#[derive(new, Clone)]
pub struct Float {
    pub val: f32,
    pub vectorization: u8,
}

#[derive(new, Clone)]
pub struct Int {
    pub val: u32,
    pub vectorization: u8,
}

#[derive(new, Clone)]
pub struct UInt {
    pub val: u32,
    pub vectorization: u8,
}

#[derive(new, Clone)]
pub struct Bool {
    pub val: bool,
    pub vectorization: u8,
}

impl RuntimeType for Float {
    type ExpandType = Arc<Variable>;
}

impl RuntimeType for Int {
    type ExpandType = Arc<Variable>;
}

impl RuntimeType for UInt {
    type ExpandType = Arc<Variable>;
}

impl RuntimeType for Bool {
    type ExpandType = Arc<Variable>;
}
