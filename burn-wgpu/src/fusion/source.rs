use crate::{
    codegen::{compiler::Compiler, dialect::gpu},
    kernel::{DynamicKernelSource, SourceTemplate},
};
use std::{marker::PhantomData, sync::Arc};

pub struct GpuKernelSource<C: Compiler> {
    pub(crate) id: String,
    pub(crate) shader: gpu::ComputeShader,
    _compiler: PhantomData<C>,
}

impl<C: Compiler> GpuKernelSource<C> {
    pub fn new(id: String, shader: gpu::ComputeShader) -> Self {
        Self {
            id,
            shader,
            _compiler: PhantomData,
        }
    }
}

impl<C: Compiler> DynamicKernelSource for Arc<GpuKernelSource<C>> {
    fn source(&self) -> SourceTemplate {
        let compiled = C::compile(self.shader.clone());
        SourceTemplate::new(compiled.to_string())
    }

    fn id(&self) -> String {
        self.id.clone()
    }
}
