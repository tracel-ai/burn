use super::dialect::gpu;
use crate::{FloatElement, IntElement};
use std::fmt::Display;

pub trait Compiler: Sync + Send + 'static {
    type Representation: Display;
    type Float: FloatElement;
    type Int: IntElement;

    fn compile(shader: gpu::ComputeShader) -> Self::Representation;
    fn elem_size(elem: gpu::Elem) -> usize;
}
