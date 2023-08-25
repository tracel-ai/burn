use alloc::vec::Vec;

use super::ParamId;
use crate::{
    record::Record,
    tensor::backend::{ADBackend, Backend},
};
pub use burn_derive::Module;
use burn_tensor::Tensor;

// At the moment, our plan is to continue experimenting with the macro internally and monitor its development.
// We may consider making it public in the future.
macro_rules! module {
    (map=$module:ident, ops=$item:expr) => {{
        struct Mapper;
        impl<B: Backend> ModuleMapper<B> for Mapper {
            fn map<const D: usize>(&mut self, _id: &ParamId, tensor: Tensor<B, D>) -> Tensor<B, D> {
                let func = $item;
                func(tensor)
            }
        }
        let mut mapper = Mapper;
        $module.map(&mut mapper)
    }};
    (map=$module:ident, ops=$item:expr, capture={$capture:ident: $ty:ty}) => {{
        struct Mapper<'a, B: Backend> {
            capture: &'a $ty,
            backend: core::marker::PhantomData<B>,
        }
        impl<'a, B: Backend> ModuleMapper<B> for Mapper<'a, B> {
            fn map<const D: usize>(&mut self, _id: &ParamId, tensor: Tensor<B, D>) -> Tensor<B, D> {
                let func = $item;
                func(tensor, self.capture)
            }
        }
        let mut mapper = Mapper {
            capture: $capture,
            backend: core::marker::PhantomData,
        };
        $module.map(&mut mapper)
    }};
    (visit=$module:ident, ops=$item:expr, state=$state_ty:ty, init=$init:expr) => {{
        struct Visitor<'a, B: Backend> {
            state: &'a mut $state_ty,
            backend: core::marker::PhantomData<B>,
        }
        impl<'a, B: Backend> ModuleVisitor<B> for Visitor<'a, B> {
            fn visit<const D: usize>(&mut self, _id: &ParamId, tensor: &Tensor<B, D>) {
                let func = $item;
                func(tensor, &mut self.state)
            }
        }
        #[allow(clippy::redundant_closure_call)]
        let mut state = $init();
        let mut visitor = Visitor {
            state: &mut state,
            backend: core::marker::PhantomData,
        };
        $module.visit(&mut visitor);
        state
    }};
}

/// Trait for all neural network modules.
///
/// Modules should be created using the [derive](burn_derive::Module) attribute.
/// This will make your module trainable, savable and loadable via
/// `state` and `load`.
///
/// # Example
///
/// A module should have a [backend](crate::tensor::backend::Backend) defined as a generic
/// parameter B. This will be used by the [derive](burn_derive::Module) attribute to generate the code
/// necessary to optimize and train the module on any backend.
///
/// ```rust
/// // Not necessary when using the burn crate directly.
/// use burn_core as burn;
///
/// use burn::{
///     nn,
///     module::Module,
///     tensor::Tensor,
///     tensor::backend::Backend,
/// };
///
/// #[derive(Module, Debug)]
/// struct MyModule<B: Backend> {
///   my_param: nn::Linear<B>,
///   my_other_field: usize,
/// }
/// ```
pub trait Module<B: Backend>: Clone + Send + Sync + core::fmt::Debug {
    /// Type to save and load the module.
    type Record: Record;

    /// Get the device list of the module and all of its sub-modules.
    fn devices(&self) -> Vec<B::Device> {
        module!(
            visit = self,
            ops = |tensor: &Tensor<B, D>, state: &mut Vec<B::Device>| {
                let device = tensor.device();
                if !state.contains(&device) {
                    state.push(device);
                }
            },
            state = Vec<B::Device>,
            init = Vec::new
        )
    }

    /// Fork the module and all of its sub-modules to the given device.
    ///
    /// # Notes
    ///
    /// This is similar to [to_device](Module::to_device), but it ensures the module will
    /// have its own autodiff graph.
    fn fork(self, device: &B::Device) -> Self {
        module!(
            map = self,
            ops = |tensor: Tensor<B, D>, device: &B::Device| {
                let is_require_grad = tensor.is_require_grad();
                let mut tensor = tensor.to_device(device).detach();

                if is_require_grad {
                    tensor = tensor.require_grad();
                }

                tensor
            },
            capture = { device: B::Device }
        )
    }

    /// Move the module and all of its sub-modules to the given device.
    ///
    /// # Warnings
    ///
    /// The device operations will be registered in the autodiff graph. Therefore, be sure to call
    /// backward only one time even if you have the same module on multiple devices. If you want to
    /// call backward multiple times, look into using [fork](Module::fork) instead.
    fn to_device(self, device: &B::Device) -> Self {
        module!(
            map = self,
            ops = |tensor: Tensor<B, D>, device: &B::Device| tensor.to_device(device),
            capture = { device: B::Device }
        )
    }

    /// Each tensor in the module tree will not require grad.
    ///
    /// # Warnings
    ///
    /// This should not be used for inference, use [valid](ADModule::valid) when using
    /// AD modules. This is mostly useful when performing partial finetuning, which is updating only
    /// a small fraction of the parameters instead of finetuning all of them.
    fn no_grad(self) -> Self {
        module!(
            map = self,
            ops = |tensor: Tensor<B, D>| tensor.set_require_grad(false)
        )
    }

    /// Get the number of parameters the module has, including all of its sub-modules.
    fn num_params(&self) -> usize {
        module!(
            visit = self,
            ops = |tensor: &Tensor<B, D>, state: &mut usize| {
                *state += tensor.shape().num_elements();
            },
            state = usize,
            init = || 0
        )
    }
    /// Visit each tensor in the module with a [visitor](ModuleVisitor).
    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V);

    /// Map each tensor in the module with a [mapper](ModuleMapper).
    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self;

    /// Load the module state from a record.
    fn load_record(self, record: Self::Record) -> Self;

    /// Convert the module into a record containing the state.
    fn into_record(self) -> Self::Record;

    #[cfg(feature = "std")]
    /// Save the module to a file using the provided [file recorder](crate::record::FileRecorder).
    ///
    /// List of supported file recorders:
    ///
    /// * [default](crate::record::DefaultFileRecorder)
    /// * [bincode](crate::record::BinFileRecorder)
    /// * [bincode compressed with gzip](crate::record::BinGzFileRecorder)
    /// * [json pretty](crate::record::PrettyJsonFileRecorder)
    /// * [json compressed with gzip](crate::record::JsonGzFileRecorder)
    /// * [named mpk](crate::record::NamedMpkFileRecorder)
    /// * [named mpk compressed with gzip](crate::record::NamedMpkGzFileRecorder)
    ///
    /// ## Notes
    ///
    /// The file extension is automatically added depending on the file recorder provided, you
    /// don't have to specify it.
    fn save_file<FR: crate::record::FileRecorder, PB: Into<std::path::PathBuf>>(
        self,
        file_path: PB,
        recorder: &FR,
    ) -> Result<(), crate::record::RecorderError> {
        let record = Self::into_record(self);
        recorder.record(record, file_path.into())
    }

    #[cfg(feature = "std")]
    /// Load the module from a file using the provided [file recorder](crate::record::FileRecorder).
    ///
    /// The recorder should be the same as the one used to save the module, see
    /// [save_file](Self::save_file).
    ///
    /// ## Notes
    ///
    /// The file extension is automatically added depending on the file recorder provided, you
    /// don't have to specify it.
    fn load_file<FR: crate::record::FileRecorder, PB: Into<std::path::PathBuf>>(
        self,
        file_path: PB,
        recorder: &FR,
    ) -> Result<Self, crate::record::RecorderError> {
        let record = recorder.load(file_path.into())?;

        Ok(self.load_record(record))
    }
}

/// Module visitor trait.
pub trait ModuleVisitor<B: Backend> {
    /// Visit a tensor in the module.
    fn visit<const D: usize>(&mut self, id: &ParamId, tensor: &Tensor<B, D>);
}

/// Module mapper trait.
pub trait ModuleMapper<B: Backend> {
    /// Map a tensor in the module.
    fn map<const D: usize>(&mut self, id: &ParamId, tensor: Tensor<B, D>) -> Tensor<B, D>;
}

/// Module with auto-differentiation backend.
pub trait ADModule<B: ADBackend>: Module<B> + Send + Sync + core::fmt::Debug {
    /// Inner module without auto-differentiation.
    type InnerModule: Module<B::InnerBackend>;

    /// Get the same module, but on the inner backend without auto-differentiation.
    fn valid(&self) -> Self::InnerModule;
}
