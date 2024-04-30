use std::marker::PhantomData;

#[cfg(feature = "template")]
use crate::template::TemplateKernel;
use crate::{
    codegen::CompilerRepresentation, gpu::WorkgroupSize, kernel::GpuComputeShaderPhase, Compiler,
};
use alloc::sync::Arc;

/// Kernel for JIT backends
///
/// Notes: by default, only Jit variant exists,
/// but users can add more kernels from source by activating the
/// template feature flag.
pub enum Kernel {
    /// A JIT GPU compute shader
    JitGpu(Box<dyn JitKernel>),
    #[cfg(feature = "template")]
    /// A kernel created from source
    Custom(Box<dyn TemplateKernel>),
}

impl Kernel {
    /// ID of the kernel, for caching
    pub fn id(&self) -> String {
        match self {
            Kernel::JitGpu(shader) => shader.id(),
            #[cfg(feature = "template")]
            Kernel::Custom(template_kernel) => template_kernel.id(),
        }
    }

    /// Source of the shader
    pub fn compile(&self) -> CompiledKernel {
        match self {
            Kernel::JitGpu(shader) => shader.compile(),
            #[cfg(feature = "template")]
            Kernel::Custom(template_kernel) => template_kernel.compile(),
        }
    }

    /// Launch information of the kernel
    pub fn launch_settings(&self) -> LaunchSettings {
        match self {
            Kernel::JitGpu(shader) => shader.launch_settings(),
            #[cfg(feature = "template")]
            Kernel::Custom(template_kernel) => template_kernel.launch_settings(),
        }
    }
}

/// A kernel, compiled in the target language
pub struct CompiledKernel {
    /// Source code of the kernel
    pub source: String,
    /// Size of a workgroup for the compiled kernel
    pub workgroup_size: WorkgroupSize,
    /// The number of bytes used by the share memory
    pub shared_mem_bytes: usize,
}

/// Information needed to launch the kernel
pub struct LaunchSettings {
    /// Layout of workgroups for the kernel
    pub workgroup: WorkGroup,
}

/// Kernel trait with the ComputeShader that will be compiled and cached based on the
/// provided id.
///
/// The kernel will be launched with the given [launch settings](LaunchSettings).
pub trait JitKernel: Send + Sync {
    /// Identifier for the kernel, used for caching kernel compilation.
    fn id(&self) -> String;
    /// Compile the kernel into source
    fn compile(&self) -> CompiledKernel;
    /// Launch settings.
    fn launch_settings(&self) -> LaunchSettings;
}

/// Implementation of the [Jit Kernel trait](JitKernel) with knowledge of its compiler
#[derive(new)]
pub struct FullCompilationPhase<C: Compiler, K: GpuComputeShaderPhase> {
    kernel: K,
    workgroup: WorkGroup,
    _compiler: PhantomData<C>,
}

impl<C: Compiler, K: GpuComputeShaderPhase> JitKernel for FullCompilationPhase<C, K> {
    fn compile(&self) -> CompiledKernel {
        let gpu_ir = self.kernel.compile();
        let workgroup_size = gpu_ir.workgroup_size;
        let lower_level_ir = C::compile(gpu_ir);
        let shared_mem_bytes = lower_level_ir.shared_memory_size();
        let source = lower_level_ir.to_string();

        CompiledKernel {
            source,
            workgroup_size,
            shared_mem_bytes,
        }
    }

    fn id(&self) -> String {
        self.kernel.id().clone()
    }

    fn launch_settings(&self) -> LaunchSettings {
        LaunchSettings {
            workgroup: self.workgroup.clone(),
        }
    }
}

impl JitKernel for Arc<dyn JitKernel> {
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

impl JitKernel for Box<dyn JitKernel> {
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
pub struct WorkGroup {
    /// Work groups for the x axis.
    pub x: u32,
    /// Work groups for the y axis.
    pub y: u32,
    /// Work groups for the z axis.
    pub z: u32,
}

impl WorkGroup {
    /// Calculate the number of invocations of a compute shader.
    pub fn num_invocations(&self) -> usize {
        (self.x * self.y * self.z) as usize
    }
}
