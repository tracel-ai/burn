use crate::{codegen::CompilerRepresentation, ir::CubeDim, Compiler, Kernel};
use alloc::sync::Arc;
use std::marker::PhantomData;

/// A kernel, compiled in the target language
pub struct CompiledKernel {
    /// Source code of the kernel
    pub source: String,
    /// Size of a workgroup for the compiled kernel
    pub cube_dim: CubeDim,
    /// The number of bytes used by the share memory
    pub shared_mem_bytes: usize,
}

/// Information needed to launch the kernel
pub struct LaunchSettings {
    /// Layout of workgroups for the kernel
    pub cube_count: CubeCount,
}

/// Kernel trait with the ComputeShader that will be compiled and cached based on the
/// provided id.
///
/// The kernel will be launched with the given [launch settings](LaunchSettings).
pub trait CubeTask: Send + Sync {
    /// Identifier for the kernel, used for caching kernel compilation.
    fn id(&self) -> String;
    /// Compile the kernel into source
    fn compile(&self) -> CompiledKernel;
    /// Launch settings.
    fn launch_settings(&self) -> LaunchSettings;
}

/// Wraps a [kernel](Kernel) with its [cube count](CubeCount) to create a [cube task](CubeTask).
#[derive(new)]
pub struct KernelTask<C: Compiler, K: Kernel> {
    kernel_definition: K,
    cube_count: CubeCount,
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

    fn launch_settings(&self) -> LaunchSettings {
        LaunchSettings {
            cube_count: self.cube_count.clone(),
        }
    }
}

impl CubeTask for Arc<dyn CubeTask> {
    fn compile(&self) -> CompiledKernel {
        self.as_ref().compile()
    }

    fn id(&self) -> String {
        self.as_ref().id()
    }

    fn launch_settings(&self) -> LaunchSettings {
        self.as_ref().launch_settings()
    }
}

impl CubeTask for Box<dyn CubeTask> {
    fn compile(&self) -> CompiledKernel {
        self.as_ref().compile()
    }

    fn id(&self) -> String {
        self.as_ref().id()
    }

    fn launch_settings(&self) -> LaunchSettings {
        self.as_ref().launch_settings()
    }
}

/// Provides launch information specifying the number of work groups to be used by a compute shader.
#[derive(new, Clone, Debug)]
pub struct CubeCount {
    /// Work groups for the x axis.
    pub x: u32,
    /// Work groups for the y axis.
    pub y: u32,
    /// Work groups for the z axis.
    pub z: u32,
}

impl CubeCount {
    /// Calculate the number of invocations of a compute shader.
    pub fn num_invocations(&self) -> usize {
        (self.x * self.y * self.z) as usize
    }
}
