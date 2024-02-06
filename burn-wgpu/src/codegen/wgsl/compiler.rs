use super::shader::WgslComputeShader;
use crate::codegen::{compiler::Compiler, ComputeShader};

pub struct WgslCompiler;

impl Compiler for WgslCompiler {
    type Representation = WgslComputeShader;

    fn compile(shader: ComputeShader) -> Self::Representation {
        From::from(shader)
    }
}
