use crate::{
    codegen::{compiler::Compiler, dialect::gpu},
    gpu::ComputeShader,
    kernel::DynamicJitKernel,
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

impl<C: Compiler> DynamicJitKernel for Arc<GpuKernelSource<C>> {
    fn to_shader(&self) -> ComputeShader {
        self.shader.clone()
    }

    fn id(&self) -> String {
        self.id.clone()
    }
}
