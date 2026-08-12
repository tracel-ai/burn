use super::SourceTemplate;
use crate::{CubeRuntime, tensor::CubeTensor};
use cubecl::{
    CompilationError, Compiler, CubeTask,
    ir::{UIntKind, metadata::Info},
    prelude::*,
};

/// Kernel source to create a [source](SourceTemplate)
pub trait KernelSource: Send + 'static + Sync {
    /// Convert to [source](SourceTemplate)
    fn source(&self) -> SourceTemplate;
    /// Identifier for the kernel, used for caching kernel compilation.
    fn id(&self) -> KernelId;
}

#[derive(new)]
/// Wraps a [kernel source](KernelSource) into a [cube task](CubeTask).
pub struct SourceKernel<K> {
    kernel_source: K,
    cube_dim: CubeDim,
}

impl<C: Compiler, K: KernelSource> CubeTask<C> for SourceKernel<K> {
    fn define(&self) -> KernelDefinition {
        // A source kernel has no expanded IR, the source text is the kernel. The definition only
        // keys the compilation cache, so the source rides along in the kernel name to keep a
        // cached artifact from outliving an edit to the template.
        let settings =
            KernelSettings::new(self.cube_dim.0, ExecutionMode::Checked, AddressType::U32);
        KernelDefinition {
            body: Scope::root(settings.clone()),
            settings,
            info: Info::default(),
        }
    }

    fn compile(
        &self,
        _definition: KernelDefinition,
        _compiler: &mut C,
        _options: &C::CompilationOptions,
    ) -> Result<CompiledKernel<C>, CompilationError> {
        let source_template = self.kernel_source.source();
        let source = source_template.complete();

        Ok(CompiledKernel {
            entrypoint_name: "main".to_string(),
            debug_name: Some(core::any::type_name::<K>()),
            source,
            cube_dim: self.cube_dim,
            debug_info: None,
            repr: None,
        })
    }
}

impl<K: KernelSource> KernelMetadata for SourceKernel<K> {
    fn id(&self) -> KernelId {
        self.kernel_source.id()
    }

    fn address_type(&self) -> ElemType {
        UIntKind::U32.into()
    }
}

/// Generates kernel source code by replacing some information using templating.
#[macro_export]
macro_rules! kernel_source {
    (
        $struct:ident,
        $file:expr
    ) => {
        /// Generated kernel from a source file.
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
pub fn build_info<R: CubeRuntime>(tensors: &[&CubeTensor<R>]) -> Vec<u32> {
    let ndims = tensors[0].meta.num_dims();
    let mut info: Vec<u32> = vec![0; tensors.len() * 2 * ndims + 1];
    info[0] = ndims as u32;

    let mut current = 1;
    for tensor in tensors.iter() {
        for d in 0..ndims {
            info[current] = tensor.meta.strides()[d] as u32;
            current += 1;
        }
    }
    for tensor in tensors.iter() {
        for d in 0..ndims {
            info[current] = tensor.meta.shape()[d] as u32;
            current += 1;
        }
    }
    info
}
