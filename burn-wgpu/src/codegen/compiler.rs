use super::dialect::gpu;
use std::fmt::Display;

pub trait Compiler {
    type Representation: Display;

    fn compile(shader: gpu::ComputeShader) -> Self::Representation;
}
