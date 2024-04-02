#[cfg(feature = "extension")]
use crate::template::SourceableKernel;
use crate::{
    gpu::ComputeShader,
    kernel::{DynamicJitKernel, StaticJitKernel},
    Compiler,
};
use alloc::sync::Arc;
use core::marker::PhantomData;

/// Kernel for JIT backends
///
/// Notes: by default, only Jit variant exists,
/// but users can add more from source by activating the
/// extension feature flag.
pub enum Kernel {
    /// A JIT GPU compute shader
    Jit(Box<dyn JitKernel>),
    #[cfg(feature = "extension")]
    /// A kernel created from source
    Custom(Box<dyn SourceableKernel>),
}

impl Kernel {
    /// ID of the kernel, for caching
    pub fn id(&self) -> String {
        match self {
            Kernel::Jit(shader) => shader.id(),
            #[cfg(feature = "extension")]
            Kernel::Custom(sourceable_kernel) => sourceable_kernel.id(),
        }
    }

    /// Launch information of the kernel
    pub fn workgroup(&self) -> WorkGroup {
        match self {
            Kernel::Jit(shader) => shader.workgroup(),
            #[cfg(feature = "extension")]
            Kernel::Custom(sourceable_kernel) => sourceable_kernel.workgroup(),
        }
    }
}

/// Kernel trait with the ComputeShader that will be compiled and cached based on the
/// provided id.
///
/// The kernel will be launched with the given [workgroup](WorkGroup).
pub trait JitKernel: Send + Sync {
    /// Convert to [shader](ComputeShader)
    fn to_shader(&self) -> ComputeShader;
    /// Convert to source as string
    fn source(&self) -> String;
    /// Identifier for the kernel, used for caching kernel compilation.
    fn id(&self) -> String;
    /// Launch information.
    fn workgroup(&self) -> WorkGroup;
}

impl JitKernel for Arc<dyn JitKernel> {
    fn to_shader(&self) -> ComputeShader {
        self.as_ref().to_shader()
    }

    fn source(&self) -> String {
        self.as_ref().source()
    }

    fn id(&self) -> String {
        self.as_ref().id()
    }

    fn workgroup(&self) -> WorkGroup {
        self.as_ref().workgroup()
    }
}

impl JitKernel for Box<dyn JitKernel> {
    fn to_shader(&self) -> ComputeShader {
        self.as_ref().to_shader()
    }

    fn source(&self) -> String {
        self.as_ref().source()
    }

    fn id(&self) -> String {
        self.as_ref().id()
    }

    fn workgroup(&self) -> WorkGroup {
        self.as_ref().workgroup()
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
/// information such as [workgroup](WorkGroup).
#[derive(new)]
pub struct DynamicKernel<K, C> {
    kernel: K,
    workgroup: WorkGroup,
    _compiler: PhantomData<C>,
}

/// Wraps a [static jit kernel](StaticJitKernel) into a [Jit kernel](JitKernel) with launch
/// information such as [workgroup](WorkGroup).
#[derive(new)]
pub struct StaticKernel<K, C> {
    workgroup: WorkGroup,
    _kernel: PhantomData<K>,
    _compiler: PhantomData<C>,
}

impl<K, C: Compiler> JitKernel for DynamicKernel<K, C>
where
    K: DynamicJitKernel + 'static,
{
    fn to_shader(&self) -> ComputeShader {
        self.kernel.to_shader()
    }

    fn source(&self) -> String {
        C::compile(self.to_shader()).to_string()
    }

    fn id(&self) -> String {
        self.kernel.id()
    }

    fn workgroup(&self) -> WorkGroup {
        self.workgroup.clone()
    }
}

impl<K, C: Compiler> JitKernel for StaticKernel<K, C>
where
    K: StaticJitKernel + 'static,
{
    fn to_shader(&self) -> ComputeShader {
        K::to_shader()
    }

    fn source(&self) -> String {
        C::compile(self.to_shader()).to_string()
    }

    fn id(&self) -> String {
        format!("{:?}", core::any::TypeId::of::<K>())
    }

    fn workgroup(&self) -> WorkGroup {
        self.workgroup.clone()
    }
}
