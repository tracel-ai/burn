use crate::{codegen::CompilerRepresentation, ir::CubeDim, Compiler, Kernel, Runtime};
use alloc::sync::Arc;
use burn_compute::server::{ComputeServer, Handle};

/// A kernel, compiled in the target language
pub struct CompiledKernel {
    /// Source code of the kernel
    pub source: String,
    /// Size of a workgroup for the compiled kernel
    pub cube_dim: CubeDim,
    /// The number of bytes used by the share memory
    pub shared_mem_bytes: usize,
}

/// Kernel trait with the ComputeShader that will be compiled and cached based on the
/// provided id.
///
/// The kernel will be launched with the given [launch settings](LaunchSettings).
pub trait CubeTask<S: ComputeServer>: Send + Sync {
    /// Identifier for the kernel, used for caching kernel compilation.
    fn id(&self) -> String;
    /// Compile the kernel into source
    fn compile(&self) -> CompiledKernel;
    /// Launch settings.
    fn cube_count(&self) -> CubeCount<S>;
}

/// Wraps a [kernel](Kernel) with its [cube count](CubeCount) to create a [cube task](CubeTask).
#[derive(new)]
pub struct KernelTask<R: Runtime, K: Kernel> {
    kernel_definition: K,
    cube_count: CubeCount<R::Server>,
}

impl<R: Runtime, K: Kernel> CubeTask<R::Server> for KernelTask<R, K> {
    fn compile(&self) -> CompiledKernel {
        let gpu_ir = self.kernel_definition.define();
        let cube_dim = gpu_ir.cube_dim;
        let lower_level_ir = R::Compiler::compile(gpu_ir);
        let shared_mem_bytes = lower_level_ir.shared_memory_size();
        let source = lower_level_ir.to_string();

        CompiledKernel {
            source,
            cube_dim,
            shared_mem_bytes,
        }
    }

    fn id(&self) -> String {
        self.kernel_definition.id().clone()
    }

    fn cube_count(&self) -> CubeCount<R::Server> {
        self.cube_count.clone()
    }
}

impl<S: ComputeServer> CubeTask<S> for Arc<dyn CubeTask<S>> {
    fn compile(&self) -> CompiledKernel {
        self.as_ref().compile()
    }

    fn id(&self) -> String {
        self.as_ref().id()
    }

    fn cube_count(&self) -> CubeCount<S> {
        self.as_ref().cube_count()
    }
}

impl<S: ComputeServer> CubeTask<S> for Box<dyn CubeTask<S>> {
    fn compile(&self) -> CompiledKernel {
        self.as_ref().compile()
    }

    fn id(&self) -> String {
        self.as_ref().id()
    }

    fn cube_count(&self) -> CubeCount<S> {
        self.as_ref().cube_count()
    }
}

/// Provides launch information specifying the number of work groups to be used by a compute shader.
pub enum CubeCount<S: ComputeServer> {
    Fixed(u32, u32, u32),
    Dynamic(Handle<S>),
}

impl<S: ComputeServer> Clone for CubeCount<S> {
    fn clone(&self) -> Self {
        match self {
            Self::Fixed(arg0, arg1, arg2) => Self::Fixed(*arg0, *arg1, *arg2),
            Self::Dynamic(handle) => Self::Dynamic(handle.clone()),
        }
    }
}
