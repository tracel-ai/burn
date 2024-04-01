#[cfg(feature = "extension")]
use crate::template::SourceableKernel;
use crate::{
    gpu::ComputeShader,
    kernel::{DynamicJitKernel, StaticJitKernel},
};
use alloc::sync::Arc;
use core::marker::PhantomData;

pub enum Kernel {
    Jit(Box<dyn JitKernel>),
    #[cfg(feature = "extension")]
    Custom(Box<dyn SourceableKernel>),
}

impl Kernel {
    pub fn id(&self) -> String {
        match self {
            Kernel::Jit(shader) => shader.id(),
            #[cfg(feature = "extension")]
            Kernel::Custom(sourceable_kernel) => sourceable_kernel.id(),
        }
    }

    pub fn workgroup(&self) -> WorkGroup {
        match self {
            Kernel::Jit(shader) => shader.workgroup(),
            #[cfg(feature = "extension")]
            Kernel::Custom(sourceable_kernel) => sourceable_kernel.workgroup(),
        }
    }
}

pub trait JitKernel: Send + Sync {
    fn to_shader(&self) -> ComputeShader;

    fn id(&self) -> String;

    fn workgroup(&self) -> WorkGroup;
}

impl JitKernel for Arc<dyn JitKernel> {
    fn to_shader(&self) -> ComputeShader {
        self.as_ref().to_shader()
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

/// Wraps a [dynamic jit kernel](DynamicJitKernel) into a [kernel](Kernel) with launch
/// information such as [workgroup](WorkGroup).
#[derive(new)]
pub struct DynamicKernel<K> {
    kernel: K,
    workgroup: WorkGroup,
}

/// Wraps a [static jit kernel](StaticJitKernel) into a [kernel](Kernel) with launch
/// information such as [workgroup](WorkGroup).
#[derive(new)]
pub struct StaticKernel<K> {
    workgroup: WorkGroup,
    _kernel: PhantomData<K>,
}

impl<K> JitKernel for DynamicKernel<K>
where
    K: DynamicJitKernel + 'static,
{
    fn to_shader(&self) -> ComputeShader {
        self.kernel.to_shader()
    }

    fn id(&self) -> String {
        self.kernel.id()
    }

    fn workgroup(&self) -> WorkGroup {
        self.workgroup.clone()
    }
}

impl<K> JitKernel for StaticKernel<K>
where
    K: StaticJitKernel + 'static,
{
    fn to_shader(&self) -> ComputeShader {
        K::to_shader()
    }

    fn id(&self) -> String {
        format!("{:?}", core::any::TypeId::of::<K>())
    }

    fn workgroup(&self) -> WorkGroup {
        self.workgroup.clone()
    }
}
