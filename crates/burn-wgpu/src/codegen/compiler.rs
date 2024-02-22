use super::dialect::gpu;
use crate::{FloatElement, IntElement};
use std::fmt::Display;

pub trait Compiler: Sync + Send + 'static + Clone + Default + core::fmt::Debug {
    type Representation: Display;
    type Float: FloatElement;
    type Int: IntElement;
    type FullPrecisionCompiler: Compiler<
        Representation = Self::Representation,
        Float = f32,
        Int = i32,
    >;

    fn compile(shader: gpu::ComputeShader) -> Self::Representation;
    fn elem_size(elem: gpu::Elem) -> usize;
}
