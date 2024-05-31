use crate::{element::JitElement, tensor::JitTensor, JitRuntime};
use burn_cube::compute::LaunchSettings;
use burn_cube::prelude::*;

use super::SourceTemplate;

/// Kernel source to create a [source](SourceTemplate)
pub trait KernelSource: Send + 'static + Sync {
    /// Convert to [source](SourceTemplate)
    fn source(&self) -> SourceTemplate;
}

#[derive(new)]
/// Wraps a [kernel source](KernelSource) into a [cube task](CubeTask) with launch
/// information.
pub struct SourceKernel<K> {
    kernel_source: K,
    cube_count: CubeCount,
    cube_dim: CubeDim,
}

impl<K> CubeTask for SourceKernel<K>
where
    K: KernelSource + 'static,
{
    fn compile(&self) -> CompiledKernel {
        let source_template = self.kernel_source.source();
        let source = source_template.complete();

        CompiledKernel {
            source,
            cube_dim: self.cube_dim,
            shared_mem_bytes: 0,
        }
    }

    fn id(&self) -> String {
        format!("{:?}", core::any::TypeId::of::<K>())
    }

    fn launch_settings(&self) -> LaunchSettings {
        LaunchSettings {
            cube_count: self.cube_count.clone(),
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

        impl $struct {
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
pub fn build_info<R: JitRuntime, E: JitElement, const D: usize>(
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
