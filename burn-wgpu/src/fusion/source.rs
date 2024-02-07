use crate::{
    codegen::{
        compiler::Compiler,
        dialect::gpu::ComputeShader,
        dialect::wgsl::{WgslCompiler, WgslComputeShader},
    },
    kernel::{DynamicKernelSource, SourceTemplate},
};
use std::sync::Arc;

pub struct DynKernelSource {
    pub(crate) id: String,
    pub(crate) shader: WgslComputeShader,
}

impl DynKernelSource {
    pub fn new(id: String, shader: ComputeShader) -> Self {
        let shader = <WgslCompiler as Compiler>::compile(shader);

        Self { id, shader }
    }
}

impl DynamicKernelSource for Arc<DynKernelSource> {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(self.shader.to_string())
    }

    fn id(&self) -> String {
        self.id.clone()
    }
}
