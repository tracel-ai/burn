use std::marker::PhantomData;

use crate::{codegen::CompilerRepresentation, ir::CubeDim, Compiler, Kernel};
use alloc::sync::Arc;
use burn_compute::server::{Binding, ComputeServer};

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
pub trait CubeTask: Send + Sync {
    /// Identifier for the kernel, used for caching kernel compilation.
    fn id(&self) -> String;
    /// Compile the kernel into source
    fn compile(&self) -> CompiledKernel;
}

/// Wraps a [kernel](Kernel) to create a [cube task](CubeTask).
#[derive(new)]
pub struct KernelTask<C: Compiler, K: Kernel> {
    kernel_definition: K,
    _compiler: PhantomData<C>,
}

impl<C: Compiler, K: Kernel> CubeTask for KernelTask<C, K> {
    fn compile(&self) -> CompiledKernel {
        let gpu_ir = self.kernel_definition.define();
        let cube_dim = gpu_ir.cube_dim;
        let lower_level_ir = C::compile(gpu_ir);
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
}

impl CubeTask for Arc<dyn CubeTask> {
    fn compile(&self) -> CompiledKernel {
        self.as_ref().compile()
    }

    fn id(&self) -> String {
        self.as_ref().id()
    }
}

impl CubeTask for Box<dyn CubeTask> {
    fn compile(&self) -> CompiledKernel {
        self.as_ref().compile()
    }

    fn id(&self) -> String {
        self.as_ref().id()
    }
}

/// Provides launch information specifying the number of work groups to be used by a compute shader.
pub enum CubeCount<S: ComputeServer> {
    /// Dispatch x,y,z work groups.
    Static(u32, u32, u32),
    /// Dispatch work groups based on the values in this buffer. The buffer should contain a u32 array [x, y, z].
    Dynamic(Binding<S>),
}

impl<S: ComputeServer> Clone for CubeCount<S> {
    fn clone(&self) -> Self {
        match self {
            Self::Static(x, y, z) => Self::Static(*x, *y, *z),
            Self::Dynamic(handle) => Self::Dynamic(handle.clone()),
        }
    }
}
