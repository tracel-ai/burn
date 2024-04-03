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
            Kernel::JitGpu(shader) => shader.source(),
            #[cfg(feature = "template")]
            Kernel::Custom(sourceable_kernel) => sourceable_kernel.source().complete(),
        }
    }

    /// Launch information of the kernel
    pub fn launch_information(&self) -> ShaderInformation {
        match self {
            Kernel::JitGpu(shader) => shader.shader_information(),
            #[cfg(feature = "template")]
            Kernel::Custom(sourceable_kernel) => sourceable_kernel.shader_information(),
        }
    }
}

/// Kernel trait with the ComputeShader that will be compiled and cached based on the
/// provided id.
///
/// The kernel will be launched with the given [shader information](ShaderInformation).
pub trait JitKernel: Send + Sync {
    /// Convert to source as string
    fn source(&self) -> String;
    /// Identifier for the kernel, used for caching kernel compilation.
    fn id(&self) -> String;
    /// Launch information.
    fn shader_information(&self) -> ShaderInformation;
}

/// Implementation of the [Jit Kernel trait](JitKernel) with knowledge of its compiler
pub struct JitGpuKernel<C: Compiler> {
    id: String,
    info: ShaderInformation,
    shader: ComputeShader,
    _compiler: PhantomData<C>,
}

impl<C: Compiler> JitGpuKernel<C> {
    /// Create a boxed [JitGpuKernel](JitGpuKernel) from a static kernel
    pub fn from_static<K: StaticJitKernel + 'static>(workgroup: WorkGroup) -> Box<Self> {
        Box::new(Self::new(
            format!("{:?}", core::any::TypeId::of::<K>()),
            K::to_shader(),
            workgroup,
        ))
    }

    /// Create a boxed [JitGpuKernel](JitGpuKernel) from a dynamic kernel
    pub fn from_dynamic<K: DynamicJitKernel + 'static>(
        kernel: K,
        workgroup: WorkGroup,
    ) -> Box<Self> {
        Box::new(Self::new(kernel.id(), kernel.to_shader(), workgroup))
    }

    fn new(id: String, shader: ComputeShader, workgroup: WorkGroup) -> Self {
        let info = ShaderInformation::new(workgroup, Some(shader.workgroup_size));
        Self {
            id,
            shader,
            info,
            _compiler: PhantomData,
        }
    }
}

impl<C: Compiler> JitKernel for JitGpuKernel<C> {
    fn source(&self) -> String {
        C::compile(self.shader.clone()).to_string()
    }

    fn id(&self) -> String {
        self.id.clone()
    }

    fn shader_information(&self) -> ShaderInformation {
        self.info.clone()
    }
}

impl JitKernel for Arc<dyn JitKernel> {
    fn source(&self) -> String {
        self.as_ref().source()
    }

    fn id(&self) -> String {
        self.as_ref().id()
    }

    fn shader_information(&self) -> ShaderInformation {
        self.as_ref().shader_information()
    }
}

impl JitKernel for Box<dyn JitKernel> {
    fn source(&self) -> String {
        self.as_ref().source()
    }

    fn id(&self) -> String {
        self.as_ref().id()
    }

    fn shader_information(&self) -> ShaderInformation {
        self.as_ref().shader_information()
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
