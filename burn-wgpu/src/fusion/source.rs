use std::sync::Arc;

use crate::{
    codegen::ComputeShader,
    kernel::{DynamicKernelSource, SourceTemplate},
};
use serde::{Deserialize, Serialize};

#[derive(new, Clone, Serialize, Deserialize)]
pub struct DynKernelSource {
    pub(crate) id: String,
    pub(crate) shader: ComputeShader,
}

impl DynamicKernelSource for Arc<DynKernelSource> {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(self.shader.to_string())
    }

    fn id(&self) -> String {
        self.id.clone()
    }
}
