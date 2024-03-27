use crate::kernel::{DynamicKernelSource, SourceTemplate, StaticKernelSource};
use alloc::sync::Arc;
use core::marker::PhantomData;

/// Kernel trait with the [source](SourceTemplate) that will be compiled and cached based on the
/// provided id.
///
/// The kernel will be launched with the given [workgroup](WorkGroup).
pub trait Kernel: 'static + Send + Sync {
    /// Source template for the kernel.
    fn source(&self) -> SourceTemplate;
    /// Identifier for the kernel, used for caching kernel compilation.
    fn id(&self) -> String;
    /// Launch information.
    fn workgroup(&self) -> WorkGroup;
}

impl Kernel for Arc<dyn Kernel> {
    fn source(&self) -> SourceTemplate {
        self.as_ref().source()
    }

    fn id(&self) -> String {
        self.as_ref().id()
    }

    fn workgroup(&self) -> WorkGroup {
        self.as_ref().workgroup()
    }
}

impl Kernel for Box<dyn Kernel> {
    fn source(&self) -> SourceTemplate {
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

/// Wraps a [dynamic kernel source](DynamicKernelSource) into a [kernel](Kernel) with launch
/// information such as [workgroup](WorkGroup).
#[derive(new)]
pub struct DynamicKernel<K> {
    kernel: K,
    workgroup: WorkGroup,
}

/// Wraps a [static kernel source](StaticKernelSource) into a [kernel](Kernel) with launch
/// information such as [workgroup](WorkGroup).
#[derive(new)]
pub struct StaticKernel<K> {
    workgroup: WorkGroup,
    _kernel: PhantomData<K>,
}

impl<K> Kernel for DynamicKernel<K>
where
    K: DynamicKernelSource + 'static,
{
    fn source(&self) -> SourceTemplate {
        self.kernel.source()
    }

    fn id(&self) -> String {
        self.kernel.id()
    }

    fn workgroup(&self) -> WorkGroup {
        self.workgroup.clone()
    }
}

impl<K> Kernel for StaticKernel<K>
where
    K: StaticKernelSource + 'static,
{
    fn source(&self) -> SourceTemplate {
        K::source()
    }

    fn id(&self) -> String {
        format!("{:?}", core::any::TypeId::of::<K>())
    }

    fn workgroup(&self) -> WorkGroup {
        self.workgroup.clone()
    }
}
