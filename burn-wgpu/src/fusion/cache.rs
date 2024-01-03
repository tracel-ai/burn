use crate::{
    codegen::ComputeShader,
    kernel::{DynamicKernelSource, SourceTemplate},
};
use hashbrown::HashMap;
use std::sync::Arc;

/// This cache ensures that the generation of the source code is only done once when the kernel is
/// executed for the first time. Following, we only include the ID in the dynamic kernel source,
/// since we rely on the compilation cache of the WGPU compute server.
///
/// If it ever causes problems, we could cache the compute shader and put it into an Arc to avoid deep
/// cloning.
#[derive(Default, Debug)]
pub struct KernelCompilationCache {
    already_compiled_ids: HashMap<String, Arc<ComputeShader>>,
}

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

impl KernelCompilationCache {
    pub fn to_state(&self) -> HashMap<String, ComputeShader> {
        let mut state = HashMap::with_capacity(self.already_compiled_ids.len());

        for (key, value) in self.already_compiled_ids.iter() {
            state.insert(key.clone(), value.as_ref().clone());
        }

        state
    }

    pub fn get(&self, id: &str) -> Option<FusedKernelSource> {
        if let Some(value) = self.already_compiled_ids.get(id) {
            return Some(FusedKernelSource::new(id.to_string(), value.clone()));
        }

        None
    }

    pub fn insert(&mut self, id: String, code: Arc<ComputeShader>) {
        self.already_compiled_ids.insert(id, code);
    }
}
