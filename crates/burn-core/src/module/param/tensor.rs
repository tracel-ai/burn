use super::reparameterization_dyn::{self, DynReparameterization};
use super::{Param, ParamId, Parameter, ParameterValue, Reparameterization};
use crate::module::{
    Content, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper, ModuleVisitor,
};
use alloc::{boxed::Box, format, string::ToString, vec::Vec};
use burn_tensor::{Bool, Device, Float, Int, Tensor, TensorData};

impl<const D: usize> super::sealed::Sealed for Tensor<D, Float> {
    fn is_active(&self) -> bool {
        Tensor::is_require_grad(self)
    }

    fn apply_reparameterization(self, reparameterization: &dyn DynReparameterization) -> Self {
        *reparameterization
            .apply_dyn(Box::new(self))
            .downcast::<Tensor<D>>()
            .expect("Reparameterization should preserve tensor rank")
    }
}
impl<const D: usize> super::sealed::Sealed for Tensor<D, Int> {
    fn is_active(&self) -> bool {
        false
    }
}
impl<const D: usize> super::sealed::Sealed for Tensor<D, Bool> {
    fn is_active(&self) -> bool {
        false
    }
}

impl<const D: usize> ParameterValue for Tensor<D, Float> {}

impl<const D: usize> Parameter for Tensor<D, Float> {
    fn is_require_grad(&self) -> bool {
        Tensor::is_require_grad(self)
    }

    fn set_require_grad(self, require_grad: bool) -> Self {
        // Parameters keep their configured training state separately from the effective tensor.
        if require_grad && !self.is_autodiff() {
            self
        } else {
            Tensor::set_require_grad(self, require_grad)
        }
    }

    fn device(&self) -> Device {
        Tensor::device(self)
    }

    fn shape(&self) -> burn_std::Shape {
        Tensor::shape(self)
    }

    fn load_to_device(self, device: &Device) -> Self {
        if self.device() != *device {
            Tensor::to_device(self, device).detach()
        } else {
            self
        }
    }
}

impl<const D: usize> ParameterValue for Tensor<D, Int> {}

impl<const D: usize> Parameter for Tensor<D, Int> {
    fn is_require_grad(&self) -> bool {
        false
    }

    fn set_require_grad(self, _require_grad: bool) -> Self {
        self
    }

    fn device(&self) -> Device {
        Tensor::device(self)
    }

    fn shape(&self) -> burn_std::Shape {
        Tensor::shape(self)
    }

    fn load_to_device(self, device: &Device) -> Self {
        if self.device() != *device {
            Tensor::to_device(self, device)
        } else {
            self
        }
    }
}

impl<const D: usize> ParameterValue for Tensor<D, Bool> {}

impl<const D: usize> Parameter for Tensor<D, Bool> {
    fn is_require_grad(&self) -> bool {
        false
    }

    fn set_require_grad(self, _require_grad: bool) -> Self {
        self
    }

    fn device(&self) -> Device {
        Tensor::device(self)
    }

    fn shape(&self) -> burn_std::Shape {
        Tensor::shape(self)
    }

    fn load_to_device(self, device: &Device) -> Self {
        if self.device() != *device {
            Tensor::to_device(self, device)
        } else {
            self
        }
    }
}

impl<const D: usize> Param<Tensor<D>> {
    /// Create a new trainable parameter from a float tensor.
    ///
    /// # Warnings
    ///
    /// We strongly recommend using [Param::uninitialized] if you are using this method to
    /// initialize parameters inside a module, since the tensor initialization will be lazy,
    /// making the loading of weights more performant.
    pub fn from_tensor(value: Tensor<D>) -> Self {
        // A plain backend can't activate gradients immediately, so record the setting explicitly
        // for a later transition to training.
        let mut param =
            Param::initialized(ParamId::new(), Parameter::set_require_grad(value, true));
        param.is_active = true;
        param
    }

    /// Create a new parameter from data.
    pub fn from_data<T>(data: T, device: &Device) -> Self
    where
        T: Into<TensorData>,
    {
        let data: TensorData = data.into();
        // A plain backend can't activate gradients immediately, so record the setting explicitly
        // for a later transition to training.
        device.memory_persistent_allocations(data, |data| {
            let value = Tensor::from_data(data, device);
            let mut param =
                Param::initialized(ParamId::new(), Parameter::set_require_grad(value, true));
            param.is_active = true;
            param
        })
    }

    /// Attach a custom or built-in reparameterization, replacing any existing one.
    pub(crate) fn with_reparameterization<R>(mut self, reparameterization: R) -> Self
    where
        R: Reparameterization,
    {
        self.reparameterization = Some(reparameterization_dyn::boxed::<R, D>(reparameterization));
        self
    }
}

impl<const D: usize> Module for Param<Tensor<D>> {
    fn visit<V: ModuleVisitor>(&self, visitor: &mut V) {
        match self.reparameterization_dyn() {
            None => visitor.visit_float(self),
            Some(reparameterization) => {
                visitor.visit_float(&self.without_reparameterization());
                visitor.enter_module(reparameterization.name(), "Reparameterization");
                reparameterization_dyn::visit(reparameterization, visitor);
                visitor.exit_module(reparameterization.name(), "Reparameterization");
            }
        }
    }

    fn map<M: ModuleMapper>(mut self, mapper: &mut M) -> Self {
        match self.reparameterization.take() {
            None => mapper.map_float(self),
            Some(reparameterization) => {
                let base = mapper.map_float(self);
                mapper.enter_module(reparameterization.name(), "Reparameterization");
                let reparameterization = reparameterization_dyn::map(reparameterization, mapper);
                mapper.exit_module(reparameterization.name(), "Reparameterization");
                base.with_dyn_reparameterization(Some(reparameterization))
            }
        }
    }

    fn to_device(mut self, device: &Device) -> Self {
        let reparameterization = self.reparameterization.take();
        let base = self.map_to_device(device, |tensor| tensor.to_device(device));
        match reparameterization {
            None => base,
            Some(reparameterization) => {
                base.with_dyn_reparameterization(Some(reparameterization.to_device_dyn(device)))
            }
        }
    }

    fn fork(mut self, device: &Device) -> Self {
        let reparameterization = self.reparameterization.take();
        let base = self.map_to_device(device, |tensor| {
            let is_require_grad = tensor.is_require_grad();
            let mut tensor = tensor.to_device(device).detach();

            if is_require_grad {
                tensor = tensor.require_grad();
            }

            tensor
        });
        match reparameterization {
            None => base,
            Some(reparameterization) => {
                base.with_dyn_reparameterization(Some(reparameterization.fork_dyn(device)))
            }
        }
    }

    fn collect_devices(&self, mut devices: Vec<Device>) -> Vec<Device> {
        let device = self.lazy_device();

        if !devices.contains(&device) {
            devices.push(device)
        }

        if let Some(reparameterization) = self.reparameterization_dyn() {
            devices = reparameterization.collect_devices_dyn(devices);
        }

        devices
    }

    fn valid(&self) -> Self {
        // Preserve whether the parameter was active, but reset the inner value's gradient state.
        // Convert the stored base and reparameterization separately so validation never
        // evaluates the effective weight or replaces a packed base with a dense one.
        let is_active = self.is_active;
        let mut param = Param::from_mapped_value(
            self.id,
            self.base().without_autodiff().set_require_grad(false),
            self.param_mapper.clone(),
        );
        param.is_active = is_active;
        param.reparameterization = self.reparameterization_dyn().map(|state| state.valid_dyn());
        param
    }

    fn materialize(self) -> Self {
        if self.reparameterization.is_none() {
            return self;
        }

        // The merged weight is a new leaf with the base's training state. Keeping the adapter
        // graph here would retain factors that are no longer exposed to the optimizer.
        let require_grad = self.base().is_require_grad();
        let value = self.val().detach().set_require_grad(require_grad);
        let is_active = self.is_active;
        let mut param = Param::from_mapped_value(self.id, value, self.param_mapper);
        param.is_active = is_active;
        param
    }

    fn train(mut self) -> Self {
        // Keep the reparameterization structure and its parameters on the autodiff backend.
        let reparameterization = self.reparameterization.take();
        // Reinstate the parameter's training state.
        let is_active = self.is_active;
        let tensor = Tensor::from_inner(self.val()).set_require_grad(is_active);
        let mut base = Param::from_mapped_value(self.id, tensor, self.param_mapper);
        base.is_active = is_active;
        match reparameterization {
            None => base,
            Some(reparameterization) => {
                base.with_dyn_reparameterization(Some(reparameterization.train_dyn()))
            }
        }
    }
}

impl<const D: usize> ModuleDisplayDefault for Param<Tensor<D>> {
    fn content(&self, content: Content) -> Option<Content> {
        let id = if content.display_settings.show_param_id() {
            format!(", id: {}", self.id)
        } else {
            "".to_string()
        };
        let string = format!(
            "ParamTensor {{rank: {D}, shape: {:?}, kind: float{id}}}",
            self.shape().as_slice()
        );
        content.add_formatted(&string).optional()
    }
}
impl<const D: usize> ModuleDisplay for Param<Tensor<D>> {}

impl<const D: usize> Module for Param<Tensor<D, Int>> {
    fn materialize(self) -> Self {
        self
    }

    fn visit<V: ModuleVisitor>(&self, visitor: &mut V) {
        visitor.visit_int(self)
    }

    fn map<M: ModuleMapper>(self, mapper: &mut M) -> Self {
        mapper.map_int(self)
    }

    fn to_device(self, device: &Device) -> Self {
        self.map_to_device(device, |tensor| tensor.to_device(device))
    }

    fn fork(self, device: &Device) -> Self {
        self.to_device(device) // Don't support autodiff.
    }

    fn collect_devices(&self, mut devices: Vec<Device>) -> Vec<Device> {
        let device = self.lazy_device();

        if !devices.contains(&device) {
            devices.push(device)
        }

        devices
    }

    fn valid(&self) -> Self {
        Param::from_mapped_value(
            self.id,
            self.val().without_autodiff(),
            self.param_mapper.clone(),
        )
    }

    fn train(self) -> Self {
        Param::from_mapped_value(self.id, Tensor::from_inner(self.val()), self.param_mapper)
    }
}

impl<const D: usize> ModuleDisplayDefault for Param<Tensor<D, Int>> {
    fn content(&self, content: Content) -> Option<Content> {
        let id = if content.display_settings.show_param_id() {
            format!(", id: {}", self.id)
        } else {
            "".to_string()
        };
        let string = format!(
            "ParamTensor {{rank: {D}, shape: {:?}, kind: int{id}}}",
            self.shape().as_slice()
        );
        content.add_formatted(&string).optional()
    }
}
impl<const D: usize> ModuleDisplay for Param<Tensor<D, Int>> {}

impl<const D: usize> Module for Param<Tensor<D, Bool>> {
    fn materialize(self) -> Self {
        self
    }

    fn visit<V: ModuleVisitor>(&self, visitor: &mut V) {
        visitor.visit_bool(self)
    }

    fn map<M: ModuleMapper>(self, mapper: &mut M) -> Self {
        mapper.map_bool(self)
    }

    fn to_device(self, device: &Device) -> Self {
        self.map_to_device(device, |tensor| tensor.to_device(device))
    }

    fn fork(self, device: &Device) -> Self {
        self.to_device(device) // Don't support autodiff.
    }

    fn collect_devices(&self, mut devices: Vec<Device>) -> Vec<Device> {
        let device = self.lazy_device();

        if !devices.contains(&device) {
            devices.push(device)
        }

        devices
    }

    fn valid(&self) -> Self {
        Param::from_mapped_value(
            self.id,
            self.val().without_autodiff(),
            self.param_mapper.clone(),
        )
    }

    fn train(self) -> Self {
        Param::from_mapped_value(self.id, Tensor::from_inner(self.val()), self.param_mapper)
    }
}

impl<const D: usize> ModuleDisplayDefault for Param<Tensor<D, Bool>> {
    fn content(&self, content: Content) -> Option<Content> {
        let id = if content.display_settings.show_param_id() {
            format!(", id: {}", self.id)
        } else {
            "".to_string()
        };

        let string = format!(
            "ParamTensor {{rank: {D}, shape: {:?}, kind: bool{id}}}",
            self.shape().as_slice()
        );
        content.add_formatted(&string).optional()
    }
}

impl<const D: usize> ModuleDisplay for Param<Tensor<D, Bool>> {}

#[cfg(all(test, feature = "std", feature = "autodiff"))]
mod tests {
    use super::*;
    use crate as burn;
    use crate::{
        module::{LoraAdapter, Module},
        test_device,
    };
    use burn_tensor::Distribution;

    #[test]
    fn validation_does_not_apply_reparameterizations() {
        #[derive(Debug, Module)]
        struct NeverApply {}

        impl Reparameterization for NeverApply {
            const NAME: &'static str = "never_apply";

            fn apply<const D: usize>(&self, _base: Tensor<D>) -> Tensor<D> {
                panic!("validation must not evaluate effective weights")
            }
        }

        let device = test_device().autodiff();
        let param = Param::from_tensor(Tensor::<2>::ones([2, 3], &device))
            .with_reparameterization(NeverApply {});
        let validation = param.valid().valid();
        assert!(!validation.base().is_autodiff());
        assert!(validation.reparameterization::<NeverApply>().is_some());
    }

    #[test]
    fn materialize_detaches_factors_and_preserves_base_state_and_layout_mapping() {
        let device = test_device().autodiff();
        let a = Param::from_tensor(Tensor::<2>::ones([2, 1], &device));
        let b = Param::from_tensor(Tensor::<2>::ones([1, 3], &device));
        let param = Param::from_tensor(Tensor::<2>::ones([2, 3], &device))
            .load_mapper(Tensor::transpose)
            .save_mapper(Tensor::transpose)
            .with_reparameterization(LoraAdapter {
                a: a.clone(),
                b: b.clone(),
                scale: 2.0,
            });

        let merged = param.clone().materialize();
        assert!(merged.adapter().is_none());
        assert!(param.adapter().is_some());
        assert_eq!(merged.id, param.id);
        assert!(merged.base().is_require_grad());
        let value = merged.base().into_data();
        value.assert_eq(&TensorData::from([[3.0f32; 3]; 2]), true);
        let saved = merged.transform_for_save().base().into_data();
        saved.assert_eq(&TensorData::from([[3.0f32; 2]; 3]), true);
        let loaded = merged
            .clone()
            .transform_for_load(Tensor::ones([3, 2], &device), merged.id);
        assert_eq!(loaded.base().dims(), [2, 3]);

        let grads = merged.val().sum().backward();
        assert!(merged.base().grad(&grads).is_some());
        assert!(param.base().grad(&grads).is_none());
        assert!(a.val().grad(&grads).is_none());
        assert!(b.val().grad(&grads).is_none());

        let validation_merged = param.valid().materialize();
        assert!(!validation_merged.base().is_autodiff());
        assert!(validation_merged.train().base().is_require_grad());
        let frozen_merged = param.freeze().materialize();
        assert!(!frozen_merged.base().is_require_grad());
        assert!(!frozen_merged.valid().train().base().is_require_grad());
    }

    #[test]
    fn qlora_validation_and_records_preserve_packed_base_until_materialization() {
        use crate::module::{Lora, QLora, Quantizer};
        use burn_tensor::quantization::{Calibration, QuantValue};

        let device = test_device().autodiff();
        let scheme = device
            .settings()
            .quantization
            .scheme
            .with_value(QuantValue::Q8S);
        let param = Param::from_tensor(Tensor::<2>::ones([4, 4], &device)).apply_qlora(QLora::new(
            Lora::new(2, 4.0),
            Quantizer::new(Calibration::MinMax, scheme),
        ));
        let inference = param.valid();
        assert_eq!(inference.base().dtype(), param.base().dtype());
        let dtype = inference.base().dtype();
        assert!(matches!(dtype, burn_tensor::DType::QFloat(_)));
        assert!(inference.adapter().is_some());
        let restored = inference.clone().train();
        assert!(!restored.base().is_require_grad());
        let adapter = restored.adapter().unwrap();
        assert!(adapter.a.val().is_require_grad());
        assert!(adapter.b.val().is_require_grad());

        let expected = inference.val().into_data();
        let loaded = param.load_record(inference.clone().into_record());
        assert_eq!(loaded.base().dtype(), inference.base().dtype());
        assert!(loaded.adapter().is_some());
        loaded.val().into_data().assert_eq(&expected, true);

        let merged = inference.materialize().materialize();
        assert!(merged.adapter().is_none());
        assert!(merged.base().dtype().is_float());
        assert!(!merged.base().is_autodiff());
        merged.val().into_data().assert_eq(&expected, true);
        let dense = Param::from_tensor(Tensor::<2>::zeros([4, 4], &test_device()));
        let loaded = dense.load_record(merged.into_record());
        loaded.val().into_data().assert_eq(&expected, true);
    }

    fn lazy_param_for_device_inspection<T: Parameter>(
        device: &Device,
        shape: [usize; 2],
    ) -> Param<T> {
        Param::uninitialized(
            ParamId::new(),
            |_, _| panic!("device inspection must not initialize parameters"),
            device.clone(),
            false,
            shape.into(),
        )
    }

    #[test]
    fn devices_preserves_lazy_parameters_of_all_tensor_kinds() {
        #[derive(Module, Debug)]
        struct MixedParameters {
            float: Param<Tensor<2>>,
            int: Param<Tensor<2, Int>>,
            bool: Param<Tensor<2, Bool>>,
            initialized: Param<Tensor<2>>,
        }

        let device = test_device();
        let model = MixedParameters {
            float: lazy_param_for_device_inspection(&device, [2, 3]),
            int: lazy_param_for_device_inspection(&device, [2, 3]),
            bool: lazy_param_for_device_inspection(&device, [2, 3]),
            initialized: Param::from_tensor(Tensor::ones([2, 3], &device)),
        };
        let clone = model.clone();

        assert_eq!(model.devices(), alloc::vec![device.clone()]);
        assert_eq!(clone.devices(), alloc::vec![device]);
        assert!(!model.float.is_initialized());
        assert!(!model.int.is_initialized());
        assert!(!model.bool.is_initialized());
    }

    #[test]
    fn devices_preserves_lazy_reparameterization_state() {
        let device = test_device();
        let param: Param<Tensor<2>> = lazy_param_for_device_inspection(&device, [2, 3]);

        // Cover both lazy and initialized bases with lazy adapter parameters.
        for base in [
            param,
            Param::from_tensor(Tensor::<2>::ones([2, 3], &device)),
        ] {
            let was_initialized = base.is_initialized();
            let param = base.with_reparameterization(LoraAdapter {
                a: lazy_param_for_device_inspection(&device, [2, 1]),
                b: lazy_param_for_device_inspection(&device, [1, 3]),
                scale: 1.0,
            });

            assert_eq!(param.devices(), alloc::vec![device.clone()]);
            assert_eq!(param.is_initialized(), was_initialized);
            let adapter = param.adapter().unwrap();
            assert!(!adapter.a.is_initialized());
            assert!(!adapter.b.is_initialized());
        }
    }

    #[test]
    fn devices_preserves_gradients_after_a_lazy_module_move() {
        let device = test_device().autodiff();
        let model = crate::test_utils::SimpleLinear {
            weight: Param::uninitialized(
                ParamId::new(),
                |device, require_grad| Tensor::ones([1, 2], device).set_require_grad(require_grad),
                device.clone(),
                true,
                [1, 2].into(),
            ),
            bias: Some(Param::uninitialized(
                ParamId::new(),
                |device, require_grad| Tensor::zeros([1], device).set_require_grad(require_grad),
                device.clone(),
                true,
                [1].into(),
            )),
        };

        assert_eq!(model.devices(), alloc::vec![device.clone()]);
        assert!(!model.weight.is_initialized());
        assert!(!model.bias.as_ref().unwrap().is_initialized());

        let model = model.to_device(&device);
        let output = Tensor::<2>::ones([1, 2], &device).matmul(model.weight.val().transpose())
            + model.bias.as_ref().unwrap().val().unsqueeze();
        let grads = output.sum().backward();

        model
            .weight
            .grad(&grads)
            .unwrap()
            .into_data()
            .assert_eq(&TensorData::from([[1.0f32, 1.0f32]]), false);
        model
            .bias
            .as_ref()
            .unwrap()
            .grad(&grads)
            .unwrap()
            .into_data()
            .assert_eq(&TensorData::from([1.0f32]), false);
    }

    #[test]
    fn set_require_grad_updates_lazy_lifecycle_state() {
        let device = test_device().autodiff();
        let param: Param<Tensor<2>> = Param::uninitialized(
            ParamId::new(),
            |device, require_grad| Tensor::ones([2, 3], device).set_require_grad(require_grad),
            device,
            true,
            [2, 3].into(),
        );

        let param = param.set_require_grad(false);

        assert!(!param.is_initialized());
        assert!(!param.is_active);
        assert!(!param.val().is_require_grad());

        let param = param.valid().train();

        assert!(!param.is_require_grad());
        assert!(!param.is_active);
    }

    #[test]
    fn set_require_grad_on_a_plain_tensor_is_applied_by_train() {
        let device = test_device();
        let param = Param::initialized(
            ParamId::new(),
            Tensor::<2>::ones([2, 3], &device).set_require_grad(false),
        )
        .set_require_grad(true);

        assert!(!param.is_require_grad());
        assert!(param.is_active);

        let param = param.train();

        assert!(param.is_require_grad());
        assert!(param.is_active);
    }

    #[test]
    fn trainable_param_created_on_a_plain_device_is_applied_by_train() {
        let device = test_device();
        let param = Param::from_tensor(Tensor::<2>::ones([2, 3], &device));

        assert!(!param.is_require_grad());
        assert!(param.is_active);

        let param = param.train();

        assert!(param.is_require_grad());
        assert!(param.is_active);
    }

    #[test]
    fn lazy_activation_setting_on_a_plain_device_is_applied_by_train() {
        let device = test_device();
        let param: Param<Tensor<2>> = Param::uninitialized(
            ParamId::new(),
            |device, require_grad| Tensor::ones([2, 3], device).set_require_grad(require_grad),
            device,
            false,
            [2, 3].into(),
        )
        .set_require_grad(true);

        assert!(!param.is_initialized());
        assert!(param.is_active);
        assert!(!param.val().is_require_grad());

        let param = param.train();

        assert!(param.is_require_grad());
        assert!(param.is_active);
    }

    #[test]
    fn mapping_a_validation_param_preserves_what_train_restores() {
        let device = test_device().autodiff();
        let param = Param::from_tensor(Tensor::<2>::ones([2, 3], &device))
            .valid()
            .map(|tensor| tensor);

        assert!(!param.is_require_grad());
        assert!(param.is_active);

        let param = param.train();

        assert!(param.is_require_grad());
        assert!(param.is_active);
    }

    #[test]
    fn mapped_value_reconstruction_preserves_what_train_restores() {
        let device = test_device().autodiff();
        let valid = Param::from_tensor(Tensor::<2>::ones([2, 3], &device)).valid();
        let (id, tensor, mapper) = valid.consume();

        let param = Param::from_mapped_value(id, tensor, mapper);

        assert!(!param.is_require_grad());
        assert!(param.is_active);

        let param = param.train();

        assert!(param.is_require_grad());
        assert!(param.is_active);
    }

    #[test]
    fn loading_a_validation_param_preserves_what_train_restores() {
        let device = test_device().autodiff();
        let valid = Param::from_tensor(Tensor::<2>::ones([2, 3], &device)).valid();
        let record = Tensor::<2>::zeros([2, 3], &test_device());

        let param = valid.transform_for_load(record, ParamId::new());

        assert!(!param.is_require_grad());
        assert!(param.is_active);

        let param = param.train();

        assert!(param.is_require_grad());
        assert!(param.is_active);
    }

    #[test]
    fn test_param_require_grad_stateful() {
        let device = test_device().autodiff();
        let tensor = Tensor::<2>::ones([3, 3], &device).require_grad();

        let param = Param::initialized(ParamId::new(), tensor);
        assert!(param.is_require_grad());
        assert!(param.is_active);

        let param = param.valid();
        assert!(!param.is_require_grad());
        assert!(param.is_active); // stateful

        let param = param.train();
        assert!(param.is_require_grad());
        assert!(param.is_active); // stateful

        let param = param.no_grad();
        assert!(!param.is_require_grad());
        assert!(!param.is_active); // stateful

        let param = param.valid();
        assert!(!param.is_require_grad()); // always
        assert!(!param.is_active); // stateful

        let param = param.train();
        assert!(!param.is_require_grad());
        assert!(!param.is_active); // stateful
    }

    #[test]
    fn a_lazy_param_with_an_init_mapper_trains_on_an_autodiff_device() {
        let device = test_device().autodiff();
        let param: Param<Tensor<2>> = Param::uninitialized(
            ParamId::new(),
            |device, require_grad| Tensor::ones([2, 3], device).set_require_grad(require_grad),
            device,
            true,
            [2, 3].into(),
        )
        .init_mapper(|tensor| tensor.mul_scalar(2.0));

        let value = param.val();
        let grads = value.clone().sum().backward();

        value
            .into_data()
            .assert_eq(&TensorData::from([[2.0f32; 3]; 2]), false);
        param
            .grad(&grads)
            .expect("the mapped value is the leaf that receives the gradient")
            .into_data()
            .assert_eq(&TensorData::from([[1.0f32; 3]; 2]), false);
    }

    #[test]
    fn counting_and_materializing_an_unadapted_lazy_param_leave_it_uninitialized() {
        let param: Param<Tensor<2>> = Param::uninitialized(
            ParamId::new(),
            |device, require_grad| Tensor::ones([2, 3], device).set_require_grad(require_grad),
            test_device(),
            true,
            [2, 3].into(),
        );

        assert_eq!(Module::num_params(&param), 6);
        assert!(!param.is_initialized());
        let id = param.id;
        let param = param.materialize().materialize();
        assert_eq!(param.id, id);
        assert!(!param.is_initialized());
    }

    #[test]
    fn a_lazy_param_moved_then_loaded_never_initializes() {
        let device = test_device();
        let param: Param<Tensor<2>> = Param::uninitialized(
            ParamId::new(),
            |_, _| panic!("the moved parameter initialized before loading"),
            device.clone(),
            false,
            [2, 3].into(),
        );

        let moved = param.to_device(&device.clone().autodiff());

        assert!(!moved.is_initialized());
        moved.transform_for_load(Tensor::ones([2, 3], &device), ParamId::new());
    }

    #[test]
    fn a_moved_lazy_param_keeps_the_autodiff_context_it_was_built_with() {
        let device = test_device();
        let lazy_ones = |device: &Device| -> Param<Tensor<2>> {
            Param::uninitialized(
                ParamId::new(),
                |device, require_grad| Tensor::ones([2, 3], device).set_require_grad(require_grad),
                device.clone(),
                true,
                [2, 3].into(),
            )
        };

        let onto_autodiff = lazy_ones(&device).to_device(&device.clone().autodiff());
        let onto_plain = lazy_ones(&device.clone().autodiff()).fork(&device);

        assert!(!onto_autodiff.lazy_device().is_autodiff());
        assert!(onto_plain.lazy_device().is_autodiff());
    }

    #[test]
    fn a_moved_param_keeps_its_reparameterization() {
        let device = test_device();
        let adapter = || LoraAdapter {
            a: Param::from_tensor(Tensor::<2>::ones([3, 1], &device)),
            b: Param::from_tensor(Tensor::<2>::zeros([1, 3], &device)),
            scale: 1.0,
        };
        let lazy: Param<Tensor<2>> = Param::uninitialized(
            ParamId::new(),
            |device, _| Tensor::ones([3, 3], device),
            device.clone(),
            false,
            [3, 3].into(),
        )
        .with_reparameterization(adapter());
        let initialized = Param::from_tensor(Tensor::<2>::ones([3, 3], &device))
            .with_reparameterization(adapter());

        let target = device.clone().autodiff();

        assert!(lazy.to_device(&target).adapter().is_some());
        assert!(initialized.fork(&target).adapter().is_some());
    }

    #[test]
    fn init_mapper_preserves_lora_after_a_clone_initializes_the_base() {
        let device = test_device();
        let param: Param<Tensor<2>> = Param::uninitialized(
            ParamId::new(),
            |device, _| Tensor::ones([3, 3], device),
            device.clone(),
            false,
            [3, 3].into(),
        )
        .with_reparameterization(crate::module::LoraAdapter {
            a: Param::from_tensor(Tensor::<2>::ones([3, 1], &device)),
            b: Param::from_tensor(Tensor::<2>::ones([1, 3], &device)),
            scale: 2.0,
        });
        let clone = param.clone();
        assert!(!param.is_initialized());
        let mapped = param.init_mapper(|value| value.mul_scalar(2.0));

        // Initialize the captured base after registering the lazy mapper, then remove sharing.
        // This exercises init_mapper's map_to_device fallback, not Module::to_device or fork.
        clone
            .val()
            .into_data()
            .assert_eq(&TensorData::from([[3.0f32; 3]; 3]), true);
        drop(clone);
        assert!(!mapped.is_initialized());

        // Map the effective LoRA value: (base + scale * A @ B) * 2 = (1 + 2) * 2 = 6.
        mapped
            .val()
            .into_data()
            .assert_eq(&TensorData::from([[6.0f32; 3]; 3]), true);
    }

    #[test]
    fn a_lazy_int_param_moved_never_initializes() {
        let device = test_device();
        let param: Param<Tensor<2, Int>> = Param::uninitialized(
            ParamId::new(),
            |_, _| panic!("the moved parameter initialized"),
            device.clone(),
            false,
            [2, 3].into(),
        );

        let param = param.to_device(&device.clone().autodiff());

        assert!(!param.is_initialized());
        assert!(!param.lazy_device().is_autodiff());
    }

    #[test]
    fn a_lazy_bool_param_moved_never_initializes() {
        let device = test_device();
        let param: Param<Tensor<2, Bool>> = Param::uninitialized(
            ParamId::new(),
            |_, _| panic!("the moved parameter initialized"),
            device.clone(),
            false,
            [2, 3].into(),
        );

        let param = param.to_device(&device.clone().autodiff());

        assert!(!param.is_initialized());
        assert!(!param.lazy_device().is_autodiff());
    }

    #[test]
    fn a_lazy_param_forked_keeps_its_gradient_requirement() {
        let device = test_device();
        let param: Param<Tensor<2>> = Param::uninitialized(
            ParamId::new(),
            |device, require_grad| Tensor::ones([2, 3], device).set_require_grad(require_grad),
            device.clone().autodiff(),
            true,
            [2, 3].into(),
        );

        let param = param.fork(&device);

        assert!(!param.is_initialized());
        assert!(param.val().is_require_grad());
    }

    #[test]
    fn a_lazy_param_with_an_init_mapper_follows_the_move() {
        let device = test_device().autodiff();
        let param: Param<Tensor<2>> = Param::uninitialized(
            ParamId::new(),
            |device, require_grad| Tensor::ones([2, 3], device).set_require_grad(require_grad),
            device.clone(),
            true,
            [2, 3].into(),
        )
        .init_mapper(|tensor| tensor.mul_scalar(2.0));

        let param = param.fork(&device);

        assert!(!param.is_initialized());
        let value = param.val();
        assert!(value.device().is_autodiff());
        assert!(value.is_require_grad());
    }

    #[test]
    fn a_lazy_param_shared_with_a_clone_initializes_before_moving() {
        let device = test_device();
        let param: Param<Tensor<2>> = Param::uninitialized(
            ParamId::new(),
            |device, _| Tensor::random([2, 3], Distribution::Default, device),
            device.clone(),
            false,
            [2, 3].into(),
        );
        let clone = param.clone();

        let moved = param.to_device(&device.autodiff());

        assert!(clone.is_initialized());
        moved
            .val()
            .into_data()
            .assert_eq(&clone.val().into_data(), true);
    }
}
