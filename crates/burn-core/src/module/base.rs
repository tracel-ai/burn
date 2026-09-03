use crate::module::{Lora, ParamGroup, QLora};

use super::{ApplyReparameterization, Flag, Param, ParamId, Quantizer, Reparameterizer};
use alloc::{
    string::{String, ToString},
    vec::Vec,
};
pub use burn_derive::Module;
use burn_tensor::{Bool, Device, Int, Tensor};

/// Type alias to `Vec<Device>` which supports `no_std` environments, but automatically using
/// the `alloc` crate.
pub type Devices = Vec<Device>;

// At the moment, our plan is to continue experimenting with the macro internally and monitor its development.
// We may consider making it public in the future.
macro_rules! module {
    (map=$module:ident, ops=$item:expr, state=$state:ident:$state_ty:ty) => {{
        struct Mapper {
            $state: $state_ty,
        }
        impl ModuleMapper for Mapper {
            fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
                let func = $item;
                func(param, &self.$state)
            }
        }
        let mut mapper = Mapper { $state };
        $module.map(&mut mapper)
    }};
    (map=$module:ident, ops=$item:expr, training=$training:expr) => {{
        struct Mapper;
        impl ModuleMapper for Mapper {
            fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
                let func = $item;
                func(param)
            }

            fn map_flag(&mut self, flag: Param<Flag>) -> Param<Flag> {
                flag.with_value($training)
            }
        }
        let mut mapper = Mapper;
        $module.map(&mut mapper)
    }};
    (map=$module:ident, ops=$item:expr, group=$group:ident, state=$state:ident:$state_ty:ty) => {{
        struct Mapper {
            pub path: Vec<String>,
            pub group: ParamGroup,
            $state: $state_ty,
        }
        impl ModuleMapper for Mapper {
            fn enter_module(&mut self, name: &str, _container_type: &str) {
                self.path.push(name.to_string());
            }

            fn exit_module(&mut self, _name: &str, _container_type: &str) {
                self.path.pop();
            }

            fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
                let path = self.path.join(".");
                if self.group.matches(&param.id, Some(&path)) {
                    let func = $item;
                    return func(param, &self.$state);
                }
                param
            }
        }
        let mut mapper = Mapper {
            path: alloc::vec![],
            group: $group,
            $state,
        };
        $module.map(&mut mapper)
    }};
    (map=$module:ident, ops=$item:expr, group=$group:ident, training=$training:expr) => {{
        struct Mapper {
            pub path: Vec<String>,
            pub group: ParamGroup,
        }
        impl ModuleMapper for Mapper {
            fn enter_module(&mut self, name: &str, _container_type: &str) {
                self.path.push(name.to_string());
            }

            fn exit_module(&mut self, _name: &str, _container_type: &str) {
                self.path.pop();
            }

            fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
                let path = self.path.join(".");
                if self.group.matches(&param.id, Some(&path)) {
                    let func = $item;
                    return func(param);
                }
                param
            }

            fn map_flag(&mut self, flag: Param<Flag>) -> Param<Flag> {
                let path = self.path.join(".");
                match self.group.matches(&flag.id, Some(&path)) {
                    true => flag.with_value($training),
                    false => flag,
                }
            }
        }
        let mut mapper = Mapper {
            path: alloc::vec![],
            group: $group,
        };
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

    /// Set whether every floating-point tensor parameter in the module tree requires gradients.
    ///
    /// Module-owned control flags are left unchanged. Use [`freeze`](Module::freeze) to disable
    /// both gradients and training behavior.
    /// On a backend without autodiff, the enabled setting is preserved and takes effect when
    /// [`train`](Module::train) transitions the module back to training.
    /// Setting this to `true` overrides selective gradient configurations such as the frozen dense
    /// base weights established by [`apply_lora`](Module::apply_lora).
    fn set_require_grad(self, require_grad: bool) -> Self {
        module!(
            map = self,
            ops = |param: Param<Tensor<D>>, require_grad: &bool| param
                .set_require_grad(*require_grad),
            state = require_grad: bool
        )
    }

    /// Set whether matched floating-point tensor parameters require gradients.
    ///
    /// Only floating-point tensor parameters within `group` are affected. Other parameter values
    /// matched by the group, including module-owned control flags, are ignored; unmatched tensor
    /// parameters are also left unchanged. This means a group created with
    /// [`ParamGroup::ids_from_module`] can select an entire subtree while this method changes only
    /// its tensor gradient state.
    /// On a backend without autodiff, the enabled setting is preserved and takes effect through
    /// [`train`](Module::train).
    ///
    /// Use [`freeze_group`](Module::freeze_group) or
    /// [`unfreeze_group`](Module::unfreeze_group) when matched training flags should be changed
    /// together with tensor gradients.
    fn set_require_grad_group(self, group: ParamGroup, require_grad: bool) -> Self {
        module!(
            map = self,
            ops = |param: Param<Tensor<D>>, require_grad: &bool| param
                .set_require_grad(*require_grad),
            group = group,
            state = require_grad: bool
        )
    }

    /// Disable gradients for every floating-point tensor parameter in the module tree.
    ///
    /// This is equivalent to `set_require_grad(false)`. Module-owned control flags are left
    /// unchanged, so dropout, noise, randomized activations and batch-normalization running
    /// statistics retain their current behavior. Use [`freeze`](Module::freeze) to disable both
    /// gradients and training behavior.
    ///
    /// # Gradient and training state
    ///
    /// This persistently disables gradient tracking, including on a backend without autodiff. A
    /// later [`train`](Module::train) keeps gradients disabled; it does not undo `no_grad`.
    ///
    /// # Warnings
    ///
    /// This should not be used for inference, use [valid](AutodiffModule::valid) when using
    /// AD modules. This is useful for partial fine-tuning when tensor gradients should be disabled
    /// while layer training behavior remains active. For example, matched batch-normalization
    /// layers continue updating their running statistics, and dropout remains enabled. Use
    /// [`freeze`](Module::freeze) when that module-owned training behavior should also stop.
    fn no_grad(self) -> Self {
        self.set_require_grad(false)
    }

    /// Freeze the whole module tree.
    ///
    /// Every floating-point tensor parameter stops requiring gradients, and every module-owned
    /// training flag is disabled. This applies both to parameterless stochastic layers such as
    /// dropout, noise and randomized activations, and to stateful layers such as batch
    /// normalization.
    ///
    /// # Gradient and training state
    ///
    /// Gradient tracking and training flags are persistently disabled. On a backend without
    /// autodiff, flags are disabled immediately, and [`train`](Module::train) keeps tensors
    /// untracked rather than undoing the freeze.
    ///
    /// # Warnings
    ///
    /// This should not be used for inference; use [`valid`](AutodiffModule::valid) with AD modules
    /// instead. `freeze` is intended for partial finetuning where a module remains on the training
    /// device but should not participate in training.
    fn freeze(self) -> Self {
        module!(
            map = self,
            ops = |param: Param<Tensor<D>>| param.set_require_grad(false),
            training = false
        )
    }

    /// Unfreeze the whole module tree.
    ///
    /// Every floating-point tensor parameter is configured to require gradients, and every
    /// module-owned training flag is set to enabled rather than restored to a prior value. This
    /// overwrites selective gradient configurations, including frozen dense base weights
    /// established by [`apply_lora`](Module::apply_lora).
    ///
    /// # Gradient and training state
    ///
    /// On a backend without autodiff, training flags are enabled immediately while gradient
    /// tracking takes effect when [`train`](Module::train) transitions the module back to the
    /// autodiff backend.
    fn unfreeze(self) -> Self {
        module!(
            map = self,
            ops = |param: Param<Tensor<D>>| param.set_require_grad(true),
            training = true
        )
    }

    /// Freeze every module-owned value in the given group, leaving the rest of the module
    /// untouched.
    ///
    /// This is the group-scoped counterpart to [`freeze`](Module::freeze): where `freeze` freezes
    /// the whole module tree, `freeze_group` clears gradient requirements on matched tensor
    /// parameters and disables matched training flags.
    ///
    /// # Gradient and training state
    ///
    /// Gradient tracking and training flags are persistently disabled for matched values. On a
    /// backend without autodiff, matched flags are disabled immediately, and
    /// [`train`](Module::train) keeps matched tensors untracked rather than undoing the group
    /// freeze.
    ///
    /// # Warnings
    ///
    /// Like [`freeze`](Module::freeze), this should not be used for inference; use
    /// [valid](AutodiffModule::valid) with AD modules instead.
    fn freeze_group(self, group: ParamGroup) -> Self {
        module!(
            map = self,
            ops = |param: Param<Tensor<D>>| param.set_require_grad(false),
            group = group,
            training = false
        )
    }

    /// Unfreeze every module-owned value in the given group, leaving the rest of the module
    /// untouched.
    ///
    /// The inverse of [freeze_group](Module::freeze_group): matched tensor parameters are configured
    /// to require gradients, and matched training flags are set to enabled rather than restored to
    /// prior values.
    ///
    /// # Gradient and training state
    ///
    /// On a backend without autodiff, matched training flags are enabled immediately while gradient
    /// tracking for matched tensors takes effect when [`train`](Module::train) transitions the
    /// module back to the autodiff backend.
    fn unfreeze_group(self, group: ParamGroup) -> Self {
        module!(
            map = self,
            ops = |param: Param<Tensor<D>>| param.set_require_grad(true),
            group = group,
            training = true
        )
    }

    /// Move the module and all of its sub-modules to the autodiff backend.
    ///
    /// # Gradient and training state
    ///
    /// This is the supported transition back to training after [`valid`](AutodiffModule::valid).
    /// It applies the latest gradient-tracking settings and training-flag values. Mappings, device
    /// moves, forks and explicit state changes on a validation module preserve or update what this
    /// method applies.
    fn train(self) -> Self
    where
        Self: AutodiffModule,
    {
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
    /// Visit each parameter and module-owned control value with a [visitor](ModuleVisitor).
    fn visit<Visitor: ModuleVisitor>(&self, visitor: &mut Visitor);

    /// Map each parameter and module-owned control value with a [mapper](ModuleMapper).
    fn map<Mapper: ModuleMapper>(self, mapper: &mut Mapper) -> Self;

    /// Quantize the weights of the module.
    fn quantize_weights(self, quantizer: &mut Quantizer) -> Self {
        self.map(quantizer)
    }

    /// Quantize the weights of the given parameter group.
    fn quantize_weights_group(self, quantizer: &mut Quantizer, group: ParamGroup) -> Self {
        quantizer.set_param_group(group);
        self.map(quantizer)
    }

    /// Attach reparameterizations using the given [`Reparameterizer`].
    ///
    /// Every floating-point parameter is passed to the reparameterizer along with its module path.
    /// The reparameterizer prepares its structural base and optionally creates the state used by
    /// [`Param::val`]. [`Lora`] is a built-in example.
    ///
    /// # Limitations
    ///
    /// Nested reparameterizations are not supported. This method should only be called on modules
    /// that don't already contain reparameterized parameters.
    fn apply_reparameterization<R>(self, reparameterizer: R) -> Self
    where
        Self: Sized,
        R: Reparameterizer,
    {
        self.map(&mut ApplyReparameterization::new(reparameterizer))
    }

    /// Attach LoRA adapters to the module's 2-D weights, freezing all base tensor parameters.
    ///
    /// The same module keeps working without any code changes; adapted weights now produce
    /// `base + scale * (a @ b)`, and only the adapter factors are trainable.
    /// Module-owned control flags are preserved, so dropout and batch normalization keep their
    /// current training behavior. Call [`freeze`](Module::freeze) before this method if the base
    /// module's control behavior and running statistics should also be frozen.
    fn apply_lora(self, lora: Lora) -> Self
    where
        Self: Sized,
    {
        self.apply_reparameterization(lora)
    }

    /// Apply QLoRA to the module: quantize the (frozen) base tensor parameters and attach trainable
    /// LoRA adapters to 2-D weights.
    ///
    /// Module-owned control flags are preserved. Call [`freeze`](Module::freeze) before this
    /// method if the base module's control behavior and running statistics should also be frozen.
    fn apply_qlora(self, qlora: QLora) -> Self
    where
        Self: Sized,
    {
        self.apply_reparameterization(qlora)
    }

    /// Collect this module's tensor parameters into a [`ModuleRecord`](crate::store::ModuleRecord).
    ///
    /// The record can be saved to a burnpack file or byte buffer and applied back with
    /// [`load_record`](Module::load_record).
    /// Module-owned control values such as [`Flag`] are runtime configuration and are not
    /// recorded; loading preserves their state and identity from the destination module.
    fn into_record(self) -> crate::store::ModuleRecord
    where
        Self: Sized,
    {
        crate::store::ModuleRecord::from_module(self, None)
    }

    /// Collect the tensor parameters `group` names into a [`ModuleRecord`](crate::store::ModuleRecord).
    ///
    /// The record of a part of the module rather than all of it — what a run that trained a
    /// group writes when the rest of the module is the checkpoint it started from, and what
    /// [`load_record`](Module::load_record) applies back over that checkpoint (with
    /// [`allow_partial`](crate::store::ModuleRecord::allow_partial), since the record holds
    /// nothing for the parameters outside the group).
    ///
    /// A parameter the group does not match is skipped before its data is read, so this never
    /// materializes the rest of the module.
    fn into_record_group(self, group: ParamGroup) -> crate::store::ModuleRecord
    where
        Self: Sized,
    {
        crate::store::ModuleRecord::from_module(self, Some(group))
    }

    /// Apply a [`ModuleRecord`](crate::store::ModuleRecord) to this module, returning the loaded
    /// module.
    ///
    /// Honors the record's [`DTypePolicy`](crate::store::DTypePolicy), `validate`, and
    /// `allow_partial` settings.
    fn try_load_record(
        self,
        record: crate::store::ModuleRecord,
    ) -> Result<Self, crate::store::RecordError>
    where
        Self: Sized,
    {
        record.apply(self)
    }

    /// Apply a [`ModuleRecord`](crate::store::ModuleRecord) to this module, consuming and returning
    /// it.
    ///
    /// Panics if validation fails; use [`try_load_record`](Module::try_load_record) for the
    /// fallible variant.
    fn load_record(self, record: crate::store::ModuleRecord) -> Self
    where
        Self: Sized,
    {
        self.try_load_record(record).expect("Failed to load record")
    }

    /// Save this module's parameters to a burnpack file on disk.
    ///
    /// Convenience for [`into_record`](Module::into_record) followed by
    /// [`ModuleRecord::save`](crate::store::ModuleRecord::save). For non-default load behavior
    /// (dtype policy, partial loading, validation), go through the record directly.
    #[cfg(feature = "std")]
    fn save_file<P: AsRef<std::path::Path>>(self, path: P) -> Result<(), crate::store::RecordError>
    where
        Self: Sized,
    {
        self.into_record().save(path)
    }

    /// Load this module's parameters from a burnpack file on disk, returning the loaded module.
    ///
    /// Uses the default load behavior. Panics on I/O or validation errors; use
    /// [`try_load_file`](Module::try_load_file) for the fallible variant, or go through
    /// [`ModuleRecord`](crate::store::ModuleRecord) to configure dtype policy, partial loading or
    /// validation.
    #[cfg(feature = "std")]
    fn load_file<P: AsRef<std::path::Path>>(self, path: P) -> Self
    where
        Self: Sized,
    {
        self.try_load_file(path)
            .expect("Failed to load module from file")
    }

    /// Fallible variant of [`load_file`](Module::load_file).
    ///
    /// Reads the record from `path` with [`ModuleRecord::load`](crate::store::ModuleRecord::load)
    /// and applies it through [`try_load_record`](Module::try_load_record).
    #[cfg(feature = "std")]
    fn try_load_file<P: AsRef<std::path::Path>>(
        self,
        path: P,
    ) -> Result<Self, crate::store::RecordError>
    where
        Self: Sized,
    {
        let record = crate::store::ModuleRecord::load(path)?;
        self.try_load_record(record)
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

    /// Visit a [`Param<Flag>`] in the module.
    ///
    /// An identified boolean value owned by a module, so traversals can reason about control state
    /// as well as tensor parameters.
    ///
    /// # Parameters
    /// - `flag`: The flag to visit
    #[allow(unused_variables)]
    fn visit_flag(&mut self, flag: &Param<Flag>) {}

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
        param
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
        param
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
        param
    }

    /// Map a [`Param<Flag>`] in the module.
    ///
    /// The default is the identity, so a mapper with no opinion about control state (for example,
    /// a record, a quantizer or a module sharder) leaves it untouched. Device moves are handled
    /// directly by [`Module::to_device`] and do not go through a [`ModuleMapper`].
    ///
    /// # Parameters
    /// - `flag`: The flag to map
    ///
    /// # Returns
    /// The mapped flag
    fn map_flag(&mut self, flag: Param<Flag>) -> Param<Flag> {
        flag
    }
}

/// Module with auto-differentiation backend.
pub trait AutodiffModule: Module + Send + core::fmt::Debug {
    /// Returns the same module on the inner backend without auto-differentiation.
    ///
    /// # Gradient and training state
    ///
    /// Tensor gradient requirements and module-owned training flags are disabled in the returned
    /// value while their configured state is retained internally. Use [`Module::train`] to apply
    /// that state. Mappings, device moves and forks preserve both the disabled effective validation
    /// state and what `train` will restore. Calling `valid` when the module's tensors are already on
    /// a plain device still disables its training flags, so the operation is idempotent but not
    /// necessarily a structural no-op.
    fn valid(&self) -> Self;

    /// Wraps an inner module back into an autodiff module and restores the training state retained
    /// by [`AutodiffModule::valid`].
    fn from_inner(module: Self) -> Self;
}

#[cfg(all(test, feature = "autodiff"))]
mod tests {
    use super::*;

    use crate::module::ParamGroup;
    use crate::{test_device, test_utils::SimpleLinear};

    fn stateful_module(device: &Device) -> (SimpleLinear, Param<Flag>) {
        (
            SimpleLinear::new(4, 4, device),
            Param::<Flag>::from_bool(true),
        )
    }

    #[test]
    fn test_module_val_train_stateful() {
        let device = test_device().autodiff();
        let module = SimpleLinear::new(4, 4, &device);

        assert!(module.weight.is_require_grad());
        assert!(module.weight.is_active);

        let module = module.valid();
        assert!(!module.weight.is_require_grad());
        assert!(module.weight.is_active); // stateful

        // Without `HasAutodiffModule`, we would need to specify the module type as well, which would be annoying
        // let module: SimpleLinear<TestAutodiffBackend> = module.train();
        let module = module.train();
        assert!(module.weight.is_require_grad());
        assert!(module.weight.is_active); // stateful

        let module = module.no_grad();
        assert!(!module.weight.is_require_grad());
        assert!(!module.weight.is_active); // stateful

        let module = module.valid();
        assert!(!module.weight.is_require_grad()); // always
        assert!(!module.weight.is_active); // stateful

        let module = module.train();
        assert!(!module.weight.is_require_grad());
        assert!(!module.weight.is_active); // stateful
    }

    /// `valid` on a module already on the inner backend returns it unchanged rather than
    /// panicking.
    #[test]
    fn valid_is_idempotent() {
        let device = test_device().autodiff();
        let module = SimpleLinear::new(4, 4, &device).valid();

        let module = module.valid();

        assert!(!module.weight.is_require_grad());
        assert!(!module.weight.val().device().is_autodiff());
    }

    #[test]
    fn valid_on_a_plain_device_keeps_tensors_plain_and_disables_flags() {
        let module = stateful_module(&test_device());

        let module = module.valid();

        assert!(!module.0.weight.val().device().is_autodiff());
        assert!(!module.1.is_enabled());
    }

    #[test]
    fn unfreeze_during_validation_is_fully_applied_by_train() {
        let device = test_device().autodiff();
        let module = stateful_module(&device).valid().unfreeze();

        // The plain validation tensor can't require gradients yet, but the setting is preserved.
        assert!(!module.0.weight.is_require_grad());
        assert!(module.0.weight.is_active);
        assert!(module.1.is_enabled());

        let module = module.train();

        assert!(module.0.weight.is_require_grad());
        assert!(module.0.bias.as_ref().unwrap().is_require_grad());
        assert!(module.1.is_enabled());
    }

    #[test]
    fn set_require_grad_during_validation_is_fully_applied_by_train() {
        let device = test_device().autodiff();
        let module = stateful_module(&device)
            .freeze()
            .valid()
            .set_require_grad(true);

        assert!(!module.0.weight.is_require_grad());
        assert!(module.0.weight.is_active);
        assert!(!module.1.is_enabled());

        let module = module.train();

        assert!(module.0.weight.is_require_grad());
        assert!(module.0.bias.as_ref().unwrap().is_require_grad());
        assert!(!module.1.is_enabled());
    }

    #[test]
    fn freeze_during_validation_survives_train() {
        let device = test_device().autodiff();
        let module = stateful_module(&device).valid().freeze().train();

        assert!(!module.0.weight.is_require_grad());
        assert!(!module.0.bias.as_ref().unwrap().is_require_grad());
        assert!(!module.1.is_enabled());
    }

    #[test]
    fn group_activation_settings_during_validation_are_applied_by_train() {
        let device = test_device().autodiff();
        let module = stateful_module(&device).freeze().valid();
        let group = ParamGroup::from_ids(vec![module.0.weight.id, module.1.id]);

        let module = module.unfreeze_group(group).train();

        assert!(module.0.weight.is_require_grad());
        assert!(!module.0.bias.as_ref().unwrap().is_require_grad());
        assert!(module.1.is_enabled());
    }

    #[test]
    fn group_gradient_settings_during_validation_ignore_flags_and_restore_tensors() {
        let device = test_device().autodiff();
        let module = stateful_module(&device).freeze().valid();
        let group = ParamGroup::from_ids(vec![module.0.weight.id, module.1.id]);

        let module = module.set_require_grad_group(group, true).train();

        assert!(module.0.weight.is_require_grad());
        assert!(!module.0.bias.as_ref().unwrap().is_require_grad());
        assert!(!module.1.is_enabled());
    }

    #[test]
    fn validation_state_survives_fork_until_train() {
        let device = test_device().autodiff();
        let module = stateful_module(&device).valid().fork(&device);

        assert!(!module.0.weight.is_require_grad());
        assert!(module.0.weight.is_active);
        assert!(!module.1.is_enabled());

        let module = module.train();

        assert!(module.0.weight.is_require_grad());
        assert!(module.0.bias.as_ref().unwrap().is_require_grad());
        assert!(module.1.is_enabled());
    }

    #[test]
    fn trained_validation_module_remains_trainable_after_fork() {
        let device = test_device().autodiff();
        let module = stateful_module(&device).valid().train().fork(&device);

        assert!(module.0.weight.is_require_grad());
        assert!(module.0.bias.as_ref().unwrap().is_require_grad());
        assert!(module.1.is_enabled());
    }

    #[test]
    fn validation_state_survives_device_move_until_train() {
        let device = test_device().autodiff();
        let target = test_device();
        let module = stateful_module(&device).valid().to_device(&target);

        assert!(!module.0.weight.is_require_grad());
        assert!(module.0.weight.is_active);
        assert!(!module.1.is_enabled());

        let module = module.train();

        assert!(module.0.weight.is_require_grad());
        assert!(module.0.bias.as_ref().unwrap().is_require_grad());
        assert!(module.1.is_enabled());
    }

    #[test]
    fn freeze_group_freezes_only_selected_params() {
        let device = test_device().autodiff();
        let module = SimpleLinear::new(4, 4, &device);

        assert!(module.weight.is_require_grad());
        assert!(module.bias.as_ref().unwrap().is_require_grad());

        let module = module.freeze_group(ParamGroup::from_path("weight"));

        assert!(!module.weight.is_require_grad());
        assert!(!module.weight.is_active);

        let bias = module.bias.as_ref().unwrap();
        assert!(bias.is_require_grad());
        assert!(bias.is_active);
    }

    #[test]
    fn unfreeze_group_only_thaws_selected_params() {
        let device = test_device().autodiff();
        let module = SimpleLinear::new(4, 4, &device);

        let module = module.no_grad();
        assert!(!module.weight.is_require_grad());
        assert!(!module.bias.as_ref().unwrap().is_require_grad());

        let module = module.unfreeze_group(ParamGroup::from_path("weight"));

        assert!(module.weight.is_require_grad());
        assert!(module.weight.is_active);
        assert!(!module.bias.as_ref().unwrap().is_require_grad());
        assert!(!module.bias.as_ref().unwrap().is_active);
    }
}
