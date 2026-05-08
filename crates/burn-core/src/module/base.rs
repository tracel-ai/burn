use super::{Param, ParamId, Quantizer};
use crate::record::Record;
use alloc::{string::String, vec::Vec};
pub use burn_derive::Module;
use burn_tensor::{Bool, Device, Int, Tensor};

/// Type alias to `Vec<Device>` which supports `no_std` environments, but automatically using
/// the `alloc` crate.
pub type Devices = Vec<Device>;

// At the moment, our plan is to continue experimenting with the macro internally and monitor its development.
// We may consider making it public in the future.
macro_rules! module {
    (map=$module:ident, ops=$item:expr) => {{
        struct Mapper;
        impl ModuleMapper for Mapper {
            fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
                let (id, tensor, mapper) = param.consume();
                let func = $item;
                let tensor = func(tensor);
                Param::from_mapped_value(id, tensor, mapper)
            }
        }
        let mut mapper = Mapper;
        $module.map(&mut mapper)
    }};
    (visit_float=$module:ident, ops=$item:expr, state=$state_ty:ty, init=$init:expr) => {{
        struct Visitor<'a> {
            state: &'a mut $state_ty,
        }
        impl<'a> ModuleVisitor for Visitor<'a> {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
                let func = $item;
                func(&param.val(), &mut self.state)
            }
        }
        #[allow(clippy::redundant_closure_call)]
        let mut state = $init();
        let mut visitor = Visitor { state: &mut state };
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
/// ```rust, ignore
/// // Not necessary when using the burn crate directly.
/// use burn_core as burn;
///
/// use burn::{
///     module::Module,
///     nn::Linear,
///     tensor::Tensor,
/// };
///
/// #[derive(Module, Debug)]
/// struct MyModule {
///   my_param: Linear,
///   my_other_field: usize,
/// }
/// ```
pub trait Module: Clone + Send + core::fmt::Debug {
    /// Type to save and load the module.
    type Record: Record;

    /// Return all the devices found in the underneath module tree added to the given vector
    /// without duplicates.
    fn collect_devices(&self, devices: Devices) -> Devices;

    /// Return all the devices found in the underneath module tree without duplicates.
    fn devices(&self) -> Devices {
        self.collect_devices(Devices::new())
    }

    /// Fork the module and all of its sub-modules to the given device.
    ///
    /// # Notes
    ///
    /// This is similar to [to_device](Module::to_device), but it ensures the output module on the
    /// new device will have its own autodiff graph.
    fn fork(self, device: &Device) -> Self;

    /// Move the module and all of its sub-modules to the given device.
    ///
    /// # Warnings
    ///
    /// The operation supports autodiff and it will be registered when activated. However, this may
    /// not be what you want. The output model will be an intermediary model, meaning that you
    /// can't optimize it with gradient descent. If you want to optimize the output network on the
    /// target device, use [fork](Module::fork) instead.
    fn to_device(self, device: &Device) -> Self;

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
            ops = |tensor: Tensor<D>| tensor.set_require_grad(false)
        )
    }

    /// Move the module and all of its sub-modules to the autodiff backend.
    ///
    /// # Notes
    ///
    /// * Only plain modules (not already on an autodiff backend) can be moved.
    /// * Calling `train()` on a module that is already on an autodiff backend
    ///   will result in a type error, because the module's inner backend does not match.
    fn train(self) -> Self
    where
        Self: AutodiffModule,
    {
        // <Self as HasAutodiffModule>::TrainModule::from_inner(self)
        AutodiffModule::from_inner(self)
    }

    /// Get the number of parameters the module has, including all of its sub-modules.
    fn num_params(&self) -> usize {
        module!(
            visit_float = self,
            ops = |tensor: &Tensor<D>, state: &mut usize| {
                *state += tensor.shape().num_elements();
            },
            state = usize,
            init = || 0
        )
    }
    /// Visit each tensor parameter in the module with a [visitor](ModuleVisitor).
    fn visit<Visitor: ModuleVisitor>(&self, visitor: &mut Visitor);

    /// Map each tensor parameter in the module with a [mapper](ModuleMapper).
    fn map<Mapper: ModuleMapper>(self, mapper: &mut Mapper) -> Self;

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
        FR: crate::record::FileRecorder,
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
        device: &Device,
    ) -> Result<Self, crate::record::RecorderError>
    where
        FR: crate::record::FileRecorder,
        PB: Into<std::path::PathBuf>,
    {
        let record = recorder.load(file_path.into(), device)?;

        Ok(self.load_record(record))
    }

    /// Quantize the weights of the module.
    fn quantize_weights(self, quantizer: &mut Quantizer) -> Self {
        self.map(quantizer)
    }
}

/// Module visitor trait for traversing and inspecting module parameters.
pub trait ModuleVisitor {
    /// Visit a float parameter in the module.
    ///
    /// # Parameters
    /// - `param`: The float parameter to visit
    #[allow(unused_variables)]
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {}

    /// Visit an int parameter in the module.
    ///
    /// # Parameters
    /// - `param`: The integer parameter to visit
    #[allow(unused_variables)]
    fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<D, Int>>) {}

    /// Visit a bool parameter in the module.
    ///
    /// # Parameters
    /// - `param`: The boolean parameter to visit
    #[allow(unused_variables)]
    fn visit_bool<const D: usize>(&mut self, param: &Param<Tensor<D, Bool>>) {}

    /// Called when entering a submodule.
    ///
    /// # Parameters
    /// - `name`: The name of the submodule being entered
    /// - `container_type`: The type of the container with format:
    ///   - For user-defined structs: "Struct:TypeName" (e.g., "Struct:Linear")
    ///   - For user-defined enums: "Enum:TypeName" (e.g., "Enum:MyEnum")
    ///   - For Vec containers: "Vec" (name is the index)
    ///   - For Tuple containers: "Tuple" (name is the index)
    ///   - For Array containers: "Array" (name is the index)
    ///
    /// Note: Option containers do not call enter_module/exit_module to preserve
    /// the field name in the path (e.g., "bias" instead of "bias.Some")
    #[allow(unused_variables)]
    fn enter_module(&mut self, name: &str, container_type: &str) {}

    /// Called when exiting a submodule.
    ///
    /// # Parameters
    /// - `name`: The name of the submodule being exited
    /// - `container_type`: The type of the container with format:
    ///   - For user-defined structs: "Struct:TypeName" (e.g., "Struct:Linear")
    ///   - For user-defined enums: "Enum:TypeName" (e.g., "Enum:MyEnum")
    ///   - For Vec containers: "Vec" (name is the index)
    ///   - For Tuple containers: "Tuple" (name is the index)
    ///   - For Array containers: "Array" (name is the index)
    ///
    /// Note: Option containers do not call enter_module/exit_module to preserve
    /// the field name in the path (e.g., "bias" instead of "bias.Some")
    #[allow(unused_variables)]
    fn exit_module(&mut self, name: &str, container_type: &str) {}

    /// Visit a float tensor with its full module path.
    ///
    /// # Parameters
    /// - `path`: The path components to the tensor as a slice (e.g., &["encoder", "layer1", "weight"]).
    ///   Each element represents a module name in the hierarchy, with the final element
    ///   being the parameter name. This allows efficient reuse of the path stack.
    /// - `id`: The unique identifier of the parameter
    /// - `tensor`: The float tensor to visit
    #[allow(unused_variables)]
    fn visit_float_with_path<const D: usize>(
        &mut self,
        path: &[String],
        id: ParamId,
        tensor: &Tensor<D>,
    ) {
    }

    /// Visit an int tensor with its full module path.
    ///
    /// # Parameters
    /// - `path`: The path components to the tensor as a slice (e.g., &["encoder", "layer1", "weight"]).
    ///   Each element represents a module name in the hierarchy, with the final element
    ///   being the parameter name. This allows efficient reuse of the path stack.
    /// - `id`: The unique identifier of the parameter
    /// - `tensor`: The integer tensor to visit
    #[allow(unused_variables)]
    fn visit_int_with_path<const D: usize>(
        &mut self,
        path: &[String],
        id: ParamId,
        tensor: &Tensor<D, Int>,
    ) {
    }

    /// Visit a bool tensor with its full module path.
    ///
    /// # Parameters
    /// - `path`: The path components to the tensor as a slice (e.g., &["encoder", "layer1", "weight"]).
    ///   Each element represents a module name in the hierarchy, with the final element
    ///   being the parameter name. This allows efficient reuse of the path stack.
    /// - `id`: The unique identifier of the parameter
    /// - `tensor`: The boolean tensor to visit
    #[allow(unused_variables)]
    fn visit_bool_with_path<const D: usize>(
        &mut self,
        path: &[String],
        id: ParamId,
        tensor: &Tensor<D, Bool>,
    ) {
    }
}

/// Module mapper trait for transforming module parameters.
pub trait ModuleMapper {
    /// Called when entering a submodule.
    ///
    /// # Parameters
    /// - `name`: The name of the submodule being entered
    /// - `container_type`: The type of the container with format:
    ///   - For user-defined structs: "Struct:TypeName" (e.g., "Struct:Linear")
    ///   - For user-defined enums: "Enum:TypeName" (e.g., "Enum:MyEnum")
    ///   - For Vec containers: "Vec" (name is the index)
    ///   - For Tuple containers: "Tuple" (name is the index)
    ///   - For Array containers: "Array" (name is the index)
    ///
    /// Note: Option containers do not call enter_module/exit_module to preserve
    /// the field name in the path (e.g., "bias" instead of "bias.Some")
    #[allow(unused_variables)]
    fn enter_module(&mut self, name: &str, container_type: &str) {}

    /// Called when exiting a submodule.
    ///
    /// # Parameters
    /// - `name`: The name of the submodule being exited
    /// - `container_type`: The type of the container with format:
    ///   - For user-defined structs: "Struct:TypeName" (e.g., "Struct:Linear")
    ///   - For user-defined enums: "Enum:TypeName" (e.g., "Enum:MyEnum")
    ///   - For Vec containers: "Vec" (name is the index)
    ///   - For Tuple containers: "Tuple" (name is the index)
    ///   - For Array containers: "Array" (name is the index)
    ///
    /// Note: Option containers do not call enter_module/exit_module to preserve
    /// the field name in the path (e.g., "bias" instead of "bias.Some")
    #[allow(unused_variables)]
    fn exit_module(&mut self, name: &str, container_type: &str) {}

    /// Map a float parameter in the module.
    ///
    /// # Parameters
    /// - `param`: The float parameter to transform
    ///
    /// # Returns
    /// The transformed parameter
    #[allow(unused_variables)]
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let (id, tensor, mapper) = param.consume();
        Param::from_mapped_value(id, tensor, mapper)
    }

    /// Map an int parameter in the module.
    ///
    /// # Parameters
    /// - `param`: The integer parameter to transform
    ///
    /// # Returns
    /// The transformed parameter
    #[allow(unused_variables)]
    fn map_int<const D: usize>(&mut self, param: Param<Tensor<D, Int>>) -> Param<Tensor<D, Int>> {
        let (id, tensor, mapper) = param.consume();
        Param::from_mapped_value(id, tensor, mapper)
    }

    /// Map a bool parameter in the module.
    ///
    /// # Parameters
    /// - `param`: The boolean parameter to transform
    ///
    /// # Returns
    /// The transformed parameter
    #[allow(unused_variables)]
    fn map_bool<const D: usize>(
        &mut self,
        param: Param<Tensor<D, Bool>>,
    ) -> Param<Tensor<D, Bool>> {
        let (id, tensor, mapper) = param.consume();
        Param::from_mapped_value(id, tensor, mapper)
    }
}

/// Module with auto-differentiation backend.
pub trait AutodiffModule: Module + Send + core::fmt::Debug {
    /// Returns the same module, but on the inner backend without auto-differentiation.
    fn valid(&self) -> Self;

    /// Wraps an inner module back into an auto-diff module.
    fn from_inner(module: Self) -> Self;
}

#[cfg(all(test, feature = "autodiff"))]
mod tests {
    use super::*;

    use crate::{TestDevice, test_utils::SimpleLinear};

    #[test]
    fn test_module_val_train_stateful() {
        let device = Device::new(TestDevice::default()).autodiff();
        let module = SimpleLinear::new(4, 4, &device);

        assert!(module.weight.is_require_grad());
        assert!(module.weight.require_grad);

        let module = module.valid();
        assert!(!module.weight.is_require_grad());
        assert!(module.weight.require_grad); // stateful

        // Without `HasAutodiffModule`, we would need to specify the module type as well, which would be annoying
        // let module: SimpleLinear<TestAutodiffBackend> = module.train();
        let module = module.train();
        assert!(module.weight.is_require_grad());
        assert!(module.weight.require_grad); // stateful

        let module = module.no_grad();
        assert!(!module.weight.is_require_grad());
        assert!(!module.weight.require_grad); // stateful

        let module = module.valid();
        assert!(!module.weight.is_require_grad()); // always
        assert!(!module.weight.require_grad); // stateful

        let module = module.train();
        assert!(!module.weight.is_require_grad());
        assert!(!module.weight.require_grad); // stateful
    }
}
