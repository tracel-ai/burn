use crate::{compute::ShaderInformation, element::JitElement, tensor::JitTensor, Runtime};

use super::SourceTemplate;

/// Static kernel source to create a [source](SourceTemplate)
pub trait StaticKernelSource: Send + 'static + Sync {
    /// Convert to [source](SourceTemplate)
    fn source() -> SourceTemplate;
}

/// Dynamic kernel source to create a [source](SourceTemplate)
pub trait DynamicKernelSource: Send + 'static + Sync {
    /// Convert to [source](SourceTemplate)
    fn source(&self) -> SourceTemplate;
}

/// Kernel trait with the [source](SourceTemplate) that will be compiled and cached based on the
/// provided id.
///
/// The kernel will be launched with the given [workgroup](WorkGroup).
pub trait SourceableKernel: 'static + Send + Sync {
    /// Convert to [source](SourceTemplate)
    fn source(&self) -> SourceTemplate;
    /// Identifier for the kernel, used for caching kernel compilation.
    fn id(&self) -> String;
    /// Launch information.
    fn shader_information(&self) -> ShaderInformation;
}

#[derive(new)]
/// Wraps a [dynamic kernel source](DynamicKernelSource) into a [kernel](SourceableKernel) with launch
/// information such as [workgroup](WorkGroup).
pub struct SourceKernel<K> {
    kernel_source: K,
    shader_information: ShaderInformation,
}

impl<K> SourceableKernel for SourceKernel<K>
where
    K: DynamicKernelSource + 'static,
{
    fn source(&self) -> SourceTemplate {
        self.kernel_source.source()
    }

    fn id(&self) -> String {
        format!("{:?}", core::any::TypeId::of::<K>())
    }

    fn shader_information(&self) -> ShaderInformation {
        self.shader_information.clone()
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

        impl $crate::template::StaticKernelSource for $struct {
            fn source() -> $crate::template::SourceTemplate {
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
