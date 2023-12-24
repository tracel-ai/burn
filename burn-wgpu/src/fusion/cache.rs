use crate::{
    codegen::ComputeShader,
    kernel::{DynamicKernelSource, SourceTemplate},
};
use hashbrown::HashSet;

/// This cache ensures that the generation of the source code is only done once when the kernel is
/// executed for the first time. Following, we only include the ID in the dynamic kernel source,
/// since we rely on the compilation cache of the WGPU compute server.
///
/// If it ever causes problems, we could cache the compute shader and put it into an Arc to avoid deep
/// cloning.
#[derive(Default, Debug)]
pub struct KernelCompilationCache {
    already_compiled_ids: HashSet<String>,
}

#[derive(new)]
pub enum FusedKernelSource {
    AlreadyCompiled { id: String },
    NewKernel { id: String, shader: ComputeShader },
}

impl DynamicKernelSource for FusedKernelSource {
    fn source(&self) -> SourceTemplate {
        match self {
            FusedKernelSource::AlreadyCompiled { id: _ } => {
                panic!("Can't get the source of an already compiled kernel.")
            }
            FusedKernelSource::NewKernel {
                id: _,
                shader: source,
            } => SourceTemplate::new(source.to_string()),
        }
    }

    fn id(&self) -> String {
        match self {
            FusedKernelSource::AlreadyCompiled { id } => id.clone(),
            FusedKernelSource::NewKernel { id, shader: _ } => id.clone(),
        }
    }
}

impl KernelCompilationCache {
    pub fn get(&self, id: &str) -> Option<FusedKernelSource> {
        if self.already_compiled_ids.contains(id) {
            return Some(FusedKernelSource::AlreadyCompiled { id: id.to_string() });
        }

        None
    }

    pub fn insert(&mut self, id: String) {
        self.already_compiled_ids.insert(id);
    }
}
