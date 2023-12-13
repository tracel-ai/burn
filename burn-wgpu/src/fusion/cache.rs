use crate::codegen::ComputeShader;
use crate::kernel::{DynamicKernelSource, SourceTemplate};
use hashbrown::HashSet;

#[derive(Default, Debug)]
pub struct KernelCache {
    items: HashSet<String>,
}

pub enum CachedComputeShader {
    Cached(String),
    Compile(String, ComputeShader),
}

impl DynamicKernelSource for CachedComputeShader {
    fn source(&self) -> SourceTemplate {
        match self {
            CachedComputeShader::Cached(_) => {
                panic!("NoSource compute shader should only be used by a higher level cache.")
            }
            CachedComputeShader::Compile(_, shader) => SourceTemplate::new(shader.to_string()),
        }
    }

    fn id(&self) -> String {
        match self {
            CachedComputeShader::Cached(id) => id.clone(),
            CachedComputeShader::Compile(id, _) => id.clone(),
        }
    }
}

impl KernelCache {
    pub fn get(&self, id: &str) -> Option<CachedComputeShader> {
        if self.items.contains(id) {
            return Some(CachedComputeShader::Cached(id.to_string()));
        }

        None
    }

    pub fn insert(&mut self, id: String) {
        self.items.insert(id);
    }
}
