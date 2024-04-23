use crate::{
    compute::{CompiledKernel, LaunchSettings, WorkGroup},
    element::JitElement,
    gpu::WorkgroupSize,
    tensor::JitTensor,
    Runtime,
};

use super::SourceTemplate;

/// Kernel source to create a [source](SourceTemplate)
pub trait KernelSource: Send + 'static + Sync {
    /// Convert to [source](SourceTemplate)
    fn source(&self) -> SourceTemplate;
}

/// Kernel trait with the [source](SourceTemplate) that will be compiled and cached based on the
/// provided id.
///
/// The kernel will be launched with the given [launch settings](LaunchSettings)
pub trait TemplateKernel: 'static + Send + Sync {
    /// Convert to source
    fn compile(&self) -> CompiledKernel;
    /// Identifier for the kernel, used for caching kernel compilation.
    fn id(&self) -> String;
    /// Launch information.
    fn launch_settings(&self) -> LaunchSettings;
}

#[derive(new)]
/// Wraps a [kernel source](KernelSource) into a [template kernel](TemplateKernel) with launch
/// information.
pub struct SourceKernel<K> {
    kernel_source: K,
    workgroup: WorkGroup,
    workgroup_size: WorkgroupSize,
}

impl<K> TemplateKernel for SourceKernel<K>
where
    K: KernelSource + 'static,
{
    fn compile(&self) -> CompiledKernel {
        let source_template = self.kernel_source.source();
        let source = source_template.complete();

        CompiledKernel {
            source,
            workgroup_size: self.workgroup_size,
            shared_mem_bytes: 0,
        }
    }

    fn id(&self) -> String {
        format!("{:?}", core::any::TypeId::of::<K>())
    }

    fn launch_settings(&self) -> LaunchSettings {
        LaunchSettings {
            workgroup: self.workgroup.clone(),
        }
    }
}

/// Generates kernel source code by replacing some information using templating.
#[macro_export]
macro_rules! kernel_wgsl {
    (
        $struct:ident,
        $file:expr
    ) => {
        /// Generated kernel from wgsl file.
        #[derive(new)]
        pub struct $struct;

        impl $crate::template::KernelSource for $struct {
            fn source(&self) -> $crate::template::SourceTemplate {
                $crate::template::SourceTemplate::new(include_str!($file))
            }
        }
    };
}

/// Create a vector containing the dimension, strides and shape of tensors.
///
/// # Example
///
/// With two tensors (lhs, rhs)
///
/// | Indexes                  | Value       |
/// |:------------------------:|:-----------:|
/// |           0..1           | D           |
/// |           1..D + 1       | lhs strides |
/// |     (D + 1)..(2 * D + 1) | rhs strides |
/// | (2 * D + 1)..(3 * D + 1) | lhs shape   |
/// | (3 * D + 1)..(4 * D + 1) | rhs shape   |
pub fn build_info<R: Runtime, E: JitElement, const D: usize>(
    tensors: &[&JitTensor<R, E, D>],
) -> Vec<u32> {
    let mut info: Vec<u32> = vec![0; tensors.len() * 2 * D + 1];
    info[0] = D as u32;

    let mut current = 1;
    for tensor in tensors.iter() {
        for d in 0..D {
            info[current] = tensor.strides[d] as u32;
            current += 1;
        }
    }
    for tensor in tensors.iter() {
        for d in 0..D {
            info[current] = tensor.shape.dims[d] as u32;
            current += 1;
        }
    }
    info
}
