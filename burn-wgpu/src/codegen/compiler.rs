use super::ComputeShader;
use std::fmt::Display;

pub trait Compiler {
    type Representation: Display;

    fn compile(shader: ComputeShader) -> Self::Representation;
}
