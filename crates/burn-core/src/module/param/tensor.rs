use super::{LoraAdapter, Param, ParamId, Parameter};
use crate::module::{
    AutodiffModule, Content, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper,
    ModuleVisitor,
};
use alloc::{boxed::Box, format, string::ToString, vec::Vec};
use burn_tensor::{Bool, Device, Float, Int, Tensor, TensorData};

impl<const D: usize> super::sealed::Sealed for Tensor<D, Float> {}
impl<const D: usize> super::sealed::Sealed for Tensor<D, Int> {}
impl<const D: usize> super::sealed::Sealed for Tensor<D, Bool> {}

impl<const D: usize> Parameter for Tensor<D, Float> {
    fn device(&self) -> Device {
        Tensor::device(self)
    }

    fn is_require_grad(&self) -> bool {
        Tensor::is_require_grad(self)
    }

    fn set_require_grad(self, require_grad: bool) -> Self {
        Tensor::set_require_grad(self, require_grad)
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

    fn compose_lora(self, adapter: &LoraAdapter) -> Self {
        // `delta` has shape `[d_in, d_out]` (rank 2); the base weight is rank `D == 2` when an
        // adapter is attached, so the reshape is an identity that only adjusts the static rank.
        let delta = adapter.delta().reshape(Tensor::shape(&self));
        self.add(delta)
    }
}

impl<const D: usize> Parameter for Tensor<D, Int> {
    fn device(&self) -> Device {
        Tensor::device(self)
    }

    fn is_require_grad(&self) -> bool {
        false
    }

    fn set_require_grad(self, _require_grad: bool) -> Self {
        self
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

impl<const D: usize> Parameter for Tensor<D, Bool> {
    fn device(&self) -> Device {
        Tensor::device(self)
    }

    fn is_require_grad(&self) -> bool {
        false
    }

    fn set_require_grad(self, _require_grad: bool) -> Self {
        self
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
}

/// Visit the trainable factors of a LoRA [adapter](LoraAdapter) as nested parameters, so the
/// record/optimizer traversal sees them at stable paths (e.g. `weight.lora.a`, `weight.lora.b`).
fn visit_adapter<V: ModuleVisitor>(adapter: &LoraAdapter, visitor: &mut V) {
    visitor.enter_module("lora", "Struct:LoraAdapter");
    visitor.enter_module("a", "Struct:LoraAdapter");
    Module::visit(&adapter.a, visitor);
    visitor.exit_module("a", "Struct:LoraAdapter");
    visitor.enter_module("b", "Struct:LoraAdapter");
    Module::visit(&adapter.b, visitor);
    visitor.exit_module("b", "Struct:LoraAdapter");
    visitor.exit_module("lora", "Struct:LoraAdapter");
}

/// Map the trainable factors of a LoRA [adapter](LoraAdapter), mirroring [`visit_adapter`] so the
/// optimizer/record mapper resolves the same paths.
fn map_adapter<M: ModuleMapper>(adapter: LoraAdapter, mapper: &mut M) -> LoraAdapter {
    let LoraAdapter { a, b, scale } = adapter;
    mapper.enter_module("lora", "Struct:LoraAdapter");
    mapper.enter_module("a", "Struct:LoraAdapter");
    let a = Module::map(a, mapper);
    mapper.exit_module("a", "Struct:LoraAdapter");
    mapper.enter_module("b", "Struct:LoraAdapter");
    let b = Module::map(b, mapper);
    mapper.exit_module("b", "Struct:LoraAdapter");
    mapper.exit_module("lora", "Struct:LoraAdapter");
    LoraAdapter { a, b, scale }
}

impl<const D: usize> Module for Param<Tensor<D>> {
    fn visit<V: ModuleVisitor>(&self, visitor: &mut V) {
        match self.adapter() {
            None => visitor.visit_float(self),
            Some(adapter) => {
                // Visit the frozen base, then the trainable adapter factors as separate leaves.
                visitor.visit_float(&self.without_adapter());
                visit_adapter(adapter, visitor);
            }
        }
    }

    fn map<M: ModuleMapper>(mut self, mapper: &mut M) -> Self {
        match self.adapter.take() {
            None => mapper.map_float(self),
            Some(adapter) => {
                // `self` no longer carries the adapter, so the mapper operates on the raw base.
                let base = mapper.map_float(self);
                let adapter = map_adapter(*adapter, mapper);
                base.with_adapter(Some(Box::new(adapter)))
            }
        }
    }

    fn to_device(mut self, device: &Device) -> Self {
        let adapter = self.adapter.take();
        let base = self.map(|tensor| tensor.to_device(device));
        match adapter {
            None => base,
            Some(adapter) => base.with_adapter(Some(Box::new(LoraAdapter {
                a: Module::to_device(adapter.a, device),
                b: Module::to_device(adapter.b, device),
                scale: adapter.scale,
            }))),
        }
    }

    fn fork(mut self, device: &Device) -> Self {
        let adapter = self.adapter.take();
        let base = self.map(|tensor| {
            let is_require_grad = tensor.is_require_grad();
            let mut tensor = tensor.to_device(device).detach();

            if is_require_grad {
                tensor = tensor.require_grad();
            }

            tensor
        });
        match adapter {
            None => base,
            Some(adapter) => base.with_adapter(Some(Box::new(LoraAdapter {
                a: Module::fork(adapter.a, device),
                b: Module::fork(adapter.b, device),
                scale: adapter.scale,
            }))),
        }
    }

    fn collect_devices(&self, mut devices: Vec<Device>) -> Vec<Device> {
        let device = self.base().device();

        if !devices.contains(&device) {
            devices.push(device)
        }

        if let Some(adapter) = self.adapter() {
            devices = Module::collect_devices(&adapter.a, devices);
            devices = Module::collect_devices(&adapter.b, devices);
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
        // Preserve initialized param `require_grad` state, but reset the inner value's.
        // When a LoRA adapter is attached, `val()` folds it into the base for inference, so the
        // resulting inference parameter is a plain (adapter-free) composed weight.
        let require_grad = self.require_grad;
        let mut param = Param::initialized(self.id, self.val().inner().set_require_grad(false));
        param.require_grad = require_grad;
        param
    }

    fn from_inner(mut module: Self) -> Self {
        // Keep the adapter structure (and its trainable factors) when moving onto the autodiff
        // backend, so the adapter remains trainable after `train()`.
        let adapter = module.adapter.take();
        // Reinstate the param's `require_grad` state
        let tensor = Tensor::from_inner(module.val()).set_require_grad(module.require_grad);
        let base = Param::initialized(module.id, tensor);
        match adapter {
            None => base,
            Some(adapter) => base.with_adapter(Some(Box::new(LoraAdapter {
                a: AutodiffModule::from_inner(adapter.a),
                b: AutodiffModule::from_inner(adapter.b),
                scale: adapter.scale,
            }))),
        }
    }
}

// impl<const D: usize, B: AutodiffBackend> HasAutodiffModule
//     for Param<Tensor<B::InnerBackend, D>>
// {
//     type TrainModule = Param<Tensor<D>>;
// }

impl<const D: usize> AutodiffModule for Param<Tensor<D, Int>> {
    fn valid(&self) -> Self {
        Param::initialized(self.id, self.val().inner())
    }

    fn from_inner(module: Self) -> Self {
        Param::initialized(module.id, Tensor::from_inner(module.val()))
    }
}

impl<const D: usize> AutodiffModule for Param<Tensor<D, Bool>> {
    fn valid(&self) -> Self {
        Param::initialized(self.id, self.val().inner())
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
        assert!(param.require_grad);

        let param = param.valid();
        assert!(!param.is_require_grad());
        assert!(param.require_grad); // stateful

        // Without `HasAutodiffModule`, we would need to specify the param type as well, which would be annoying:
        // let param: Param<Tensor<TestAutodiffBackend, _>> = param.train();
        let param = param.train();
        assert!(param.is_require_grad());
        assert!(param.require_grad); // stateful

        let param = param.no_grad();
        assert!(!param.is_require_grad());
        assert!(!param.require_grad); // stateful

        let param = param.valid();
        assert!(!param.is_require_grad()); // always
        assert!(!param.require_grad); // stateful

        let param = param.train();
        assert!(!param.is_require_grad());
        assert!(!param.require_grad); // stateful
    }
}
