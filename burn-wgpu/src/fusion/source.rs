use crate::{
    codegen::{compiler::Compiler, dialect::gpu::ComputeShader, dialect::wgsl::WgslComputeShader},
    kernel::{DynamicKernelSource, SourceTemplate},
};
use std::sync::Arc;

pub struct GpuKernelSource {
    pub(crate) id: String,
    pub(crate) shader: WgslComputeShader,
}

impl GpuKernelSource {
    pub fn new<C>(id: String, shader: ComputeShader) -> Self
    where
        C: Compiler<Representation = WgslComputeShader>,
    {
        let shader = C::compile(shader);

        Self { id, shader }
    }
}

impl DynamicKernelSource for Arc<GpuKernelSource> {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(self.shader.to_string())
    }

    fn id(&self) -> String {
        self.id.clone()
    }
}
