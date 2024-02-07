use super::shader::WgslComputeShader;
use crate::codegen::{compiler::Compiler, dialect::gpu};

pub struct WgslCompiler;

impl Compiler for WgslCompiler {
    type Representation = WgslComputeShader;

    fn compile(shader: gpu::ComputeShader) -> Self::Representation {
        From::from(shader)
    }
}
