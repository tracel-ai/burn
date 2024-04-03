use std::marker::PhantomData;

#[cfg(feature = "template")]
use crate::template::SourceableKernel;
use crate::{
    gpu::{ComputeShader, WorkgroupSize},
    kernel::{DynamicJitKernel, StaticJitKernel},
    Compiler,
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
    Custom(Box<dyn SourceableKernel>),
}

impl Kernel {
    /// ID of the kernel, for caching
    pub fn id(&self) -> String {
        match self {
            Kernel::JitGpu(shader) => shader.id(),
            #[cfg(feature = "template")]
            Kernel::Custom(sourceable_kernel) => sourceable_kernel.id(),
        }
    }

    /// Source of the shader, as string
    pub fn source(&self) -> String {
        match self {
            Kernel::JitGpu(shader) => shader.compile(),
            #[cfg(feature = "template")]
            Kernel::Custom(sourceable_kernel) => sourceable_kernel.source().complete(),
        }
    }

    /// Launch information of the kernel
    pub fn launch_information(&self) -> ShaderInformation {
        match self {
            Kernel::JitGpu(shader) => shader.workgroup(),
            #[cfg(feature = "template")]
            Kernel::Custom(sourceable_kernel) => sourceable_kernel.shader_information(),
        }
    }
}

pub struct CompiledKernel {
    pub source: String,
    pub workgroup_size: WorkgroupSize,
}

pub struct LaunchSettings {
    pub workgroup: WorkGroup,
}

/// Kernel trait with the ComputeShader that will be compiled and cached based on the
/// provided id.
///
/// The kernel will be launched with the given [shader information](ShaderInformation).
pub trait JitKernel: Send + Sync {
    /// Identifier for the kernel, used for caching kernel compilation.
    fn id(&self) -> String;
    /// TODO:
    fn compile(&self) -> CompiledKernel;
    /// Launch settings.
    fn launch_settings(&self) -> LaunchSettings;
}

/// Implementation of the [Jit Kernel trait](JitKernel) with knowledge of its compiler
#[derive(new)]
pub struct DynamicJitGpuKernel<C: Compiler, K: DynamicJitKernel> {
    id: String,
    info: ShaderInformation,
    kernel: K,
    _compiler: PhantomData<C>,
}

/// Implementation of the [Jit Kernel trait](JitKernel) with knowledge of its compiler
#[derive(new)]
pub struct StaticJitGpuKernel<C: Compiler, K: StaticJitKernel> {
    workgroup: WorkGroup,
    _kernel: PhantomData<K>,
    _compiler: PhantomData<C>,
}

impl<C: Compiler, K: DynamicJitKernel> JitKernel for DynamicJitGpuKernel<C, K> {
    fn compile(&self) -> String {
        let gpu_ir = self.kernel.compile();
        let lower_level_ir = C::compile(gpu_ir);

        lower_level_ir.to_string()
    }

    fn id(&self) -> String {
        self.id.clone()
    }

    fn workgroup(&self) -> ShaderInformation {
        self.info.clone()
    }
}

impl<C: Compiler, K: StaticJitKernel> JitKernel for StaticJitGpuKernel<C, K> {
    fn compile(&self) -> String {
        let gpu_ir = K::compile();
        let lower_level_ir = C::compile(gpu_ir);

        lower_level_ir.to_string()
    }

    fn id(&self) -> String {
        format!("{:?}", core::any::TypeId::of::<Self>())
    }

    fn workgroup(&self) -> ShaderInformation {
        self.info.clone()
    }
}

impl JitKernel for Arc<dyn JitKernel> {
    fn compile(&self) -> String {
        self.as_ref().compile()
    }

    fn id(&self) -> String {
        self.as_ref().id()
    }

    fn workgroup(&self) -> ShaderInformation {
        self.as_ref().workgroup()
    }
}

impl JitKernel for Box<dyn JitKernel> {
    fn compile(&self) -> String {
        self.as_ref().compile()
    }

    fn id(&self) -> String {
        self.as_ref().id()
    }

    fn workgroup(&self) -> ShaderInformation {
        self.as_ref().workgroup()
    }
}

#[derive(new, Clone)]
/// Launch information for a shader
pub struct ShaderInformation {
    /// Number of workgroups
    pub workgroup: WorkGroup,
    /// Size of a workgroup. Necessary for some runtimes
    pub workgroup_size: Option<WorkgroupSize>,
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
