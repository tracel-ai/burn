#[cfg(feature = "template")]
use crate::template::SourceableKernel;
use crate::{
    gpu::{ComputeShader, WorkgroupSize},
    kernel::{DynamicJitKernel, StaticJitKernel},
    Compiler,
};
use alloc::sync::Arc;
use core::marker::PhantomData;

/// Kernel for JIT backends
///
/// Notes: by default, only Jit variant exists,
/// but users can add more kernels from source by activating the
/// template feature flag.
pub enum Kernel {
    /// A JIT GPU compute shader
    Jit(Box<dyn JitKernel>),
    #[cfg(feature = "template")]
    /// A kernel created from source
    Custom(Box<dyn SourceableKernel>),
}

impl Kernel {
    /// ID of the kernel, for caching
    pub fn id(&self) -> String {
        match self {
            Kernel::Jit(shader) => shader.id(),
            #[cfg(feature = "template")]
            Kernel::Custom(sourceable_kernel) => sourceable_kernel.id(),
        }
    }

    /// Source of the shader, as string
    pub fn source(&self) -> String {
        match self {
            Kernel::Jit(shader) => shader.source(),
            #[cfg(feature = "template")]
            Kernel::Custom(sourceable_kernel) => sourceable_kernel.source().complete(),
        }
    }

    /// Launch information of the kernel
    pub fn launch_information(&self) -> ShaderInformation {
        match self {
            Kernel::Jit(shader) => shader.shader_information(),
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

/// Wraps a [dynamic jit kernel](DynamicJitKernel) into a [Jit kernel](JitKernel) with launch
/// information
pub struct DynamicKernel<K, C> {
    shader: ComputeShader,
    id: String,
    workgroup: WorkGroup,
    _kernel: PhantomData<K>,
    _compiler: PhantomData<C>,
}

impl<K: DynamicJitKernel + 'static, C: Compiler> DynamicKernel<K, C> {
    /// Create a dynamic kernel
    pub fn new(kernel: K, workgroup: WorkGroup) -> Self {
        Self {
            shader: kernel.to_shader(),
            id: kernel.id(),
            workgroup,
            _kernel: PhantomData,
            _compiler: PhantomData,
        }
    }
}

impl<K: DynamicJitKernel + 'static, C: Compiler> JitKernel for DynamicKernel<K, C> {
    fn source(&self) -> String {
        C::compile(self.shader.clone()).to_string()
    }

    fn id(&self) -> String {
        self.id.clone()
    }

    fn shader_information(&self) -> ShaderInformation {
        ShaderInformation::new(self.workgroup.clone(), Some(self.shader.workgroup_size))
    }
}

/// Wraps a [dynamic jit kernel](DynamicJitKernel) into a [Jit kernel](JitKernel) with launch
/// information
pub struct StaticKernel<K, C> {
    shader: ComputeShader,
    id: String,
    workgroup: WorkGroup,
    _kernel: PhantomData<K>,
    _compiler: PhantomData<C>,
}

impl<K: StaticJitKernel + 'static, C: Compiler> StaticKernel<K, C> {
    /// Create a static kernel
    pub fn new(workgroup: WorkGroup) -> Self {
        Self {
            shader: K::to_shader(),
            id: format!("{:?}", core::any::TypeId::of::<K>()),
            workgroup,
            _kernel: PhantomData,
            _compiler: PhantomData,
        }
    }
}

impl<K: StaticJitKernel + 'static, C: Compiler> JitKernel for StaticKernel<K, C> {
    fn source(&self) -> String {
        C::compile(self.shader.clone()).to_string()
    }

    fn id(&self) -> String {
        self.id.clone()
    }

    fn shader_information(&self) -> ShaderInformation {
        ShaderInformation::new(self.workgroup.clone(), Some(self.shader.workgroup_size))
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
