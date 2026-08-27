use super::reparameterization_dyn::{self, DynReparameterization};
use super::{Param, ParamId, Parameter, ParameterValue, Reparameterization};
use crate::module::{
    AutodiffModule, Content, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper,
    ModuleVisitor,
};
use alloc::{boxed::Box, format, string::ToString, vec::Vec};
use burn_tensor::{Bool, Device, Float, Int, Tensor, TensorData};

impl<const D: usize> super::sealed::Sealed for Tensor<D, Float> {
    fn is_active(&self) -> bool {
        Tensor::is_require_grad(self)
    }

    fn materialize(self, reparameterization: &dyn DynReparameterization) -> Self {
        *reparameterization
            .materialize_dyn(Box::new(self))
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
        Tensor::set_require_grad(self, require_grad)
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
    /// Create a new parameter from a float tensor.
    ///
    /// # Warnings
    ///
    /// We strongly recommend using [Param::uninitialized] if you are using this method to
    /// initialize parameters inside a module, since the tensor initialization will be lazy,
    /// making the loading of weights more performant.
    pub fn from_tensor(value: Tensor<D>) -> Self {
        // When creating a parameter from a float tensor, we automatically mark it as requiring
        // gradients, so that it can be updated by an optimizer.
        Param::initialized(ParamId::new(), value.require_grad())
    }

    /// Create a new parameter from data.
    pub fn from_data<T>(data: T, device: &Device) -> Self
    where
        T: Into<TensorData>,
    {
        let data: TensorData = data.into();
        // When creating a parameter from a float tensor, we automatically mark it as requiring
        // gradients, so that it can be updated by an optimizer.
        device.memory_persistent_allocations(data, |data| {
            let value = Tensor::from_data(data, device);
            Param::initialized(ParamId::new(), value.require_grad())
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
        let base = self.map(|tensor| tensor.to_device(device));
        match reparameterization {
            None => base,
            Some(reparameterization) => {
                base.with_dyn_reparameterization(Some(reparameterization.to_device_dyn(device)))
            }
        }
    }

    fn fork(mut self, device: &Device) -> Self {
        let reparameterization = self.reparameterization.take();
        let base = self.map(|tensor| {
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
        let device = self.base().device();

        if !devices.contains(&device) {
            devices.push(device)
        }

        if let Some(reparameterization) = self.reparameterization_dyn() {
            devices = reparameterization.collect_devices_dyn(devices);
        }

        devices
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
    fn visit<V: ModuleVisitor>(&self, visitor: &mut V) {
        visitor.visit_int(self)
    }

    fn map<M: ModuleMapper>(self, mapper: &mut M) -> Self {
        mapper.map_int(self)
    }

    fn to_device(self, device: &Device) -> Self {
        self.map(|tensor| tensor.to_device(device))
    }

    fn fork(self, device: &Device) -> Self {
        self.to_device(device) // Don't support autodiff.
    }

    fn collect_devices(&self, mut devices: Vec<Device>) -> Vec<Device> {
        let device = self.val().device();

        if !devices.contains(&device) {
            devices.push(device)
        }

        devices
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
    fn visit<V: ModuleVisitor>(&self, visitor: &mut V) {
        visitor.visit_bool(self)
    }

    fn map<M: ModuleMapper>(self, mapper: &mut M) -> Self {
        mapper.map_bool(self)
    }

    fn to_device(self, device: &Device) -> Self {
        self.map(|tensor| tensor.to_device(device))
    }

    fn fork(self, device: &Device) -> Self {
        self.to_device(device) // Don't support autodiff.
    }

    fn collect_devices(&self, mut devices: Vec<Device>) -> Vec<Device> {
        let device = self.val().device();

        if !devices.contains(&device) {
            devices.push(device)
        }

        devices
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

impl<const D: usize> AutodiffModule for Param<Tensor<D>> {
    fn valid(&self) -> Self {
        // Preserve whether the parameter was active, but reset the inner value's gradient state.
        // `val()` folds any reparameterization into the base for inference.
        let is_active = self.is_active;
        let mut param = Param::initialized(self.id, self.val().no_grad().set_require_grad(false));
        param.is_active = is_active;
        param
    }

    fn from_inner(mut module: Self) -> Self {
        // Keep the reparameterization structure and its parameters on the autodiff backend.
        let reparameterization = module.reparameterization.take();
        // Reinstate the parameter's training state.
        let tensor = Tensor::from_inner(module.val()).set_require_grad(module.is_active);
        let base = Param::initialized(module.id, tensor);
        match reparameterization {
            None => base,
            Some(reparameterization) => {
                base.with_dyn_reparameterization(Some(reparameterization.from_inner_dyn()))
            }
        }
    }
}

impl<const D: usize> AutodiffModule for Param<Tensor<D, Int>> {
    fn valid(&self) -> Self {
        Param::initialized(self.id, self.val().no_grad())
    }

    fn from_inner(module: Self) -> Self {
        Param::initialized(module.id, Tensor::from_inner(module.val()))
    }
}

impl<const D: usize> AutodiffModule for Param<Tensor<D, Bool>> {
    fn valid(&self) -> Self {
        Param::initialized(self.id, self.val().no_grad())
    }

    fn from_inner(module: Self) -> Self {
        Param::initialized(module.id, Tensor::from_inner(module.val()))
    }
}

#[cfg(all(test, feature = "std", feature = "autodiff"))]
mod tests {
    use super::*;
    use crate::{module::Module, test_device};

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

        // Without `HasAutodiffModule`, we would need to specify the param type as well, which would be annoying:
        // let param: Param<Tensor<TestAutodiffBackend, _>> = param.train();
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
}
