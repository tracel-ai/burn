use super::ParamId;
use crate::{
    record::Record,
    tensor::backend::{AutodiffBackend, Backend},
};
use alloc::vec::Vec;
pub use burn_derive::Module;
use burn_tensor::{Bool, Int, Tensor};

/// Type alias to `Vec<B::Device>` which supports `no_std` environments, but automatically using
/// the `alloc` crate.
pub type Devices<B> = Vec<<B as Backend>::Device>;

// At the moment, our plan is to continue experimenting with the macro internally and monitor its development.
// We may consider making it public in the future.
macro_rules! module {
    (map=$module:ident, ops=$item:expr) => {{
        struct Mapper;
        impl<B: Backend> ModuleMapper<B> for Mapper {
            fn map_float<const D: usize>(
                &mut self,
                _id: &ParamId,
                tensor: Tensor<B, D>,
            ) -> Tensor<B, D> {
                let func = $item;
                func(tensor)
            }
        }
        let mut mapper = Mapper;
        $module.map(&mut mapper)
    }};
    (visit_float=$module:ident, ops=$item:expr, state=$state_ty:ty, init=$init:expr) => {{
        struct Visitor<'a, B: Backend> {
            state: &'a mut $state_ty,
            backend: core::marker::PhantomData<B>,
        }
        impl<'a, B: Backend> ModuleVisitor<B> for Visitor<'a, B> {
            fn visit_float<const D: usize>(&mut self, _id: &ParamId, tensor: &Tensor<B, D>) {
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
/// ```no_run
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
pub trait Module<B: Backend>: Clone + Send + core::fmt::Debug {
    /// Type to save and load the module.
    type Record: Record<B>;

    /// Return all the devices found in the underneath module tree added to the given vector
    /// without duplicates.
    fn collect_devices(&self, devices: Devices<B>) -> Devices<B>;

    /// Return all the devices found in the underneath module tree without duplicates.
    fn devices(&self) -> Devices<B> {
        self.collect_devices(Devices::<B>::new())
    }

    /// Fork the module and all of its sub-modules to the given device.
    ///
    /// # Notes
    ///
    /// This is similar to [to_device](Module::to_device), but it ensures the output module on the
    /// new device will have its own autodiff graph.
    fn fork(self, device: &B::Device) -> Self;

    /// Move the module and all of its sub-modules to the given device.
    ///
    /// # Warnings
    ///
    /// The operation supports autodiff and it will be registered when activated. However, this may
    /// not be what you want. The output model will be an intermediary model, meaning that you
    /// can't optimize it with gradient descent. If you want to optimize the output network on the
    /// target device, use [fork](Module::fork) instead.
    fn to_device(self, device: &B::Device) -> Self;

    /// Each tensor in the module tree will not require grad.
    ///
    /// # Warnings
    ///
    /// This should not be used for inference, use [valid](AutodiffModule::valid) when using
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
            visit_float = self,
            ops = |tensor: &Tensor<B, D>, state: &mut usize| {
                *state += tensor.shape().num_elements();
            },
            state = usize,
            init = || 0
        )
    }
    /// Visit each tensor parameter in the module with a [visitor](ModuleVisitor).
    fn visit<Visitor: ModuleVisitor<B>>(&self, visitor: &mut Visitor);

    /// Map each tensor parameter in the module with a [mapper](ModuleMapper).
    fn map<Mapper: ModuleMapper<B>>(self, mapper: &mut Mapper) -> Self;

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
    fn save_file<FR, PB>(
        self,
        file_path: PB,
        recorder: &FR,
    ) -> Result<(), crate::record::RecorderError>
    where
        FR: crate::record::FileRecorder<B>,
        PB: Into<std::path::PathBuf>,
    {
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
    fn load_file<FR, PB>(
        self,
        file_path: PB,
        recorder: &FR,
        device: &B::Device,
    ) -> Result<Self, crate::record::RecorderError>
    where
        FR: crate::record::FileRecorder<B>,
        PB: Into<std::path::PathBuf>,
    {
        let record = recorder.load(file_path.into(), device)?;

        Ok(self.load_record(record))
    }
}

/// Module visitor trait.
pub trait ModuleVisitor<B: Backend> {
    /// Visit a float tensor in the module.
    fn visit_float<const D: usize>(&mut self, _id: &ParamId, _tensor: &Tensor<B, D>) {}
    /// Visit an int tensor in the module.
    fn visit_int<const D: usize>(&mut self, _id: &ParamId, _tensor: &Tensor<B, D, Int>) {}
    /// Visit a bool tensor in the module.
    fn visit_bool<const D: usize>(&mut self, _id: &ParamId, _tensor: &Tensor<B, D, Bool>) {}
}

/// Module mapper trait.
pub trait ModuleMapper<B: Backend> {
    /// Map a float tensor in the module.
    fn map_float<const D: usize>(&mut self, _id: &ParamId, tensor: Tensor<B, D>) -> Tensor<B, D> {
        tensor
    }
    /// Map an int tensor in the module.
    fn map_int<const D: usize>(
        &mut self,
        _id: &ParamId,
        tensor: Tensor<B, D, Int>,
    ) -> Tensor<B, D, Int> {
        tensor
    }
    /// Map a bool tensor in the module.
    fn map_bool<const D: usize>(
        &mut self,
        _id: &ParamId,
        tensor: Tensor<B, D, Bool>,
    ) -> Tensor<B, D, Bool> {
        tensor
    }
}

/// Module with auto-differentiation backend.
pub trait AutodiffModule<B: AutodiffBackend>: Module<B> + Send + core::fmt::Debug {
    /// Inner module without auto-differentiation.
    type InnerModule: Module<B::InnerBackend>;

    /// Get the same module, but on the inner backend without auto-differentiation.
    fn valid(&self) -> Self::InnerModule;
}
