use crate::{
    codegen::ComputeShader,
    kernel::{DynamicKernelSource, SourceTemplate},
};
use std::sync::Arc;

#[derive(new, Clone)]
pub struct FusedKernelSource {
    id: String,
    pub(crate) shader: Arc<ComputeShader>,
}

impl DynamicKernelSource for FusedKernelSource {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(self.shader.to_string())
    }

    fn id(&self) -> String {
        self.id.clone()
    }
}
