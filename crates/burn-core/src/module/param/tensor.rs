use super::{Param, ParamId, Parameter};
use crate::module::{
    AutodiffModule, Content, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper,
    ModuleVisitor,
};
use crate::tensor::{
    Tensor,
    backend::{AutodiffBackend, Backend},
};
use alloc::{format, string::ToString, vec::Vec};
use burn_tensor::{Bool, Float, Int, TensorData, ops::Device};

impl<B: Backend, const D: usize> Parameter for Tensor<B, D, Float> {
    type Device = B::Device;

    fn device(&self) -> Self::Device {
        Tensor::device(self)
    }

    fn is_require_grad(&self) -> bool {
        Tensor::is_require_grad(self)
    }

    fn set_require_grad(self, require_grad: bool) -> Self {
        Tensor::set_require_grad(self, require_grad)
    }
}

impl<B: Backend, const D: usize> Parameter for Tensor<B, D, Int> {
    type Device = B::Device;

    fn device(&self) -> Self::Device {
        Tensor::device(self)
    }

    fn is_require_grad(&self) -> bool {
        false
    }

    fn set_require_grad(self, _require_grad: bool) -> Self {
        self
    }
}

impl<B: Backend, const D: usize> Parameter for Tensor<B, D, Bool> {
    type Device = B::Device;

    fn device(&self) -> Self::Device {
        Tensor::device(self)
    }

    fn is_require_grad(&self) -> bool {
        false
    }

    fn set_require_grad(self, _require_grad: bool) -> Self {
        self
    }
}

impl<B: Backend, const D: usize> Param<Tensor<B, D>> {
    /// Create a new parameter from a float tensor.
    ///
    /// # Warnings
    ///
    /// We strongly recommend using [Param::uninitialized] if you are using this method to
    /// initialize parameters inside a module, since the tensor initialization will be lazy,
    /// making the loading of weights more performant.
    pub fn from_tensor(value: Tensor<B, D>) -> Self {
        // When creating a parameter from a float tensor, we automatically mark it as requiring
        // gradients, so that it can be updated by an optimizer.
        Param::initialized(ParamId::new(), value.require_grad())
    }

    /// The shape of the parameter, **without triggering initialization**.
    ///
    /// This is critical for shape validation during loading: when applying tensors to an
    /// uninitialized parameter, we need to validate the shape without triggering the
    /// initialization function (which would allocate an unnecessary tensor).
    ///
    /// Use this instead of [crate::tensor::Tensor::shape] when you need the shape but want to
    /// preserve lazy initialization.
    pub fn lazy_shape(&self) -> burn_tensor::Shape {
        let initialization = match &self.initialization {
            Some(init) => init,
            None => return self.shape(),
        };

        let init = initialization.read().unwrap();

        match init.as_ref() {
            Some(value) => value.shape.clone(),
            None => self.shape(),
        }
    }

    /// Create a new parameter from data.
    pub fn from_data<T>(data: T, device: &B::Device) -> Self
    where
        T: Into<TensorData>,
    {
        // When creating a parameter from a float tensor, we automatically mark it as requiring
        // gradients, so that it can be updated by an optimizer.
        B::memory_persistent_allocations(device, data, |data| {
            let value = Tensor::from_data(data, device);
            Param::initialized(ParamId::new(), value.require_grad())
        })
    }

    /// Transform a parameter for loading by applying load transformations.
    ///
    /// This method is used to restore a parameter from a tensor (typically during deserialization).
    /// It ensures the tensor is moved to the expected device, applies the param mapper's
    /// `on_load` transformation, and preserves the autodiff settings (require_grad).
    pub fn transform_for_load(self, tensor: Tensor<B, D>, param_id: ParamId) -> Self {
        let mut new_tensor = tensor;

        let mapper = self.param_mapper.clone();

        let expected_device = self.lazy_device();
        let expected_require_grad = self.lazy_is_require_grad();

        // Make sure we load the tensor into the same module device.
        if new_tensor.device() != expected_device {
            new_tensor = new_tensor.to_device(&expected_device).detach();
        }

        new_tensor = mapper.on_load(new_tensor);

        // Make sure we load the tensor with the same autodiff setting.
        new_tensor = new_tensor.set_require_grad(expected_require_grad);

        let mut loaded = Self::initialized(param_id, new_tensor);
        loaded.param_mapper = mapper;
        loaded
    }

    /// Transform a parameter for saving by applying save transformations.
    ///
    /// This method is used to prepare a parameter for saving (typically during serialization).
    /// It applies the param mapper's `on_save` transformation, which can be used
    /// to modify the tensor before serialization (e.g., quantization, precision conversion).
    pub fn transform_for_save(&self) -> Self {
        let mut tensor = self.val();
        let mapper = self.param_mapper.clone();

        tensor = mapper.on_save(tensor);

        Self::initialized(self.id, tensor)
    }
}

impl<B: Backend, const D: usize> Param<Tensor<B, D, Int>> {
    /// The shape of the parameter, **without triggering initialization**.
    ///
    /// This is critical for shape validation during loading: when applying tensors to an
    /// uninitialized parameter, we need to validate the shape without triggering the
    /// initialization function (which would allocate an unnecessary tensor).
    ///
    /// Use this instead of [crate::tensor::Tensor::shape] when you need the shape but want to
    /// preserve lazy initialization.
    pub fn lazy_shape(&self) -> burn_tensor::Shape {
        let initialization = match &self.initialization {
            Some(init) => init,
            None => return self.shape(),
        };

        let init = initialization.read().unwrap();

        match init.as_ref() {
            Some(value) => value.shape.clone(),
            None => self.shape(),
        }
    }

    /// Transform a parameter for loading by applying load transformations.
    ///
    /// This method is used to restore a parameter from a tensor (typically during deserialization).
    /// It ensures the tensor is moved to the expected device and applies the param mapper's
    /// `on_load` transformation.
    pub fn transform_for_load(self, tensor: Tensor<B, D, Int>, param_id: ParamId) -> Self {
        let mut new_tensor = tensor;

        let mapper = self.param_mapper.clone();

        let expected_device = self.lazy_device();

        // Make sure we load the tensor into the same module device.
        if new_tensor.device() != expected_device {
            new_tensor = new_tensor.to_device(&expected_device);
        }

        new_tensor = mapper.on_load(new_tensor);

        let mut loaded = Self::initialized(param_id, new_tensor);
        loaded.param_mapper = mapper;
        loaded
    }

    /// Transform a parameter for saving by applying save transformations.
    ///
    /// This method is used to prepare a parameter for saving (typically during serialization).
    /// It applies the param mapper's `on_save` transformation, which can be used
    /// to modify the tensor before serialization (e.g., quantization, precision conversion).
    pub fn transform_for_save(&self) -> Self {
        let mut tensor = self.val();
        let mapper = self.param_mapper.clone();

        tensor = mapper.on_save(tensor);

        Self::initialized(self.id, tensor)
    }
}

impl<B: Backend, const D: usize> Param<Tensor<B, D, Bool>> {
    /// The shape of the parameter, **without triggering initialization**.
    ///
    /// This is critical for shape validation during loading: when applying tensors to an
    /// uninitialized parameter, we need to validate the shape without triggering the
    /// initialization function (which would allocate an unnecessary tensor).
    ///
    /// **Returns:**
    /// - For uninitialized params: the shape from the `Uninitialized` struct
    /// - For initialized params: the actual shape from the tensor
    ///
    /// Use this instead of [crate::tensor::Tensor::shape] when you need the shape but want to
    /// preserve lazy initialization.
    pub fn lazy_shape(&self) -> burn_tensor::Shape {
        let initialization = match &self.initialization {
            Some(init) => init,
            None => return self.shape(),
        };

        let init = initialization.read().unwrap();

        match init.as_ref() {
            Some(value) => value.shape.clone(),
            None => self.shape(),
        }
    }

    /// Transform a parameter for loading by applying load transformations.
    ///
    /// This method is used to restore a parameter from a tensor (typically during deserialization).
    /// It ensures the tensor is moved to the expected device and applies the param mapper's
    /// `on_load` transformation.
    pub fn transform_for_load(self, tensor: Tensor<B, D, Bool>, param_id: ParamId) -> Self {
        let mut new_tensor = tensor;

        let mapper = self.param_mapper.clone();

        let expected_device = self.lazy_device();

        // Make sure we load the tensor into the same module device.
        if new_tensor.device() != expected_device {
            new_tensor = new_tensor.to_device(&expected_device);
        }

        new_tensor = mapper.on_load(new_tensor);

        let mut loaded = Self::initialized(param_id, new_tensor);
        loaded.param_mapper = mapper;
        loaded
    }

    /// Transform a parameter for saving by applying save transformations.
    ///
    /// This method is used to prepare a parameter for saving (typically during serialization).
    /// It applies the param mapper's `on_save` transformation, which can be used
    /// to modify the tensor before serialization (e.g., quantization, precision conversion).
    pub fn transform_for_save(&self) -> Self {
        let mut tensor = self.val();
        let mapper = self.param_mapper.clone();

        tensor = mapper.on_save(tensor);

        Self::initialized(self.id, tensor)
    }
}

impl<const D: usize, B: Backend> Module<B> for Param<Tensor<B, D>> {
    type Record = Param<Tensor<B, D>>;

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        visitor.visit_float(self)
    }

    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        mapper.map_float(self)
    }

    fn into_record(self) -> Self::Record {
        self.transform_for_save()
    }

    fn load_record(self, record: Self::Record) -> Self {
        let (record_param_id, record_tensor, _) = record.consume();
        self.transform_for_load(record_tensor, record_param_id)
    }

    fn to_device(self, device: &Device<B>) -> Self {
        self.map(|tensor| tensor.to_device(device))
    }

    fn fork(self, device: &Device<B>) -> Self {
        self.map(|tensor| {
            let device_here = tensor.device();
            let is_require_grad = tensor.is_require_grad();
            println!(
                "[{:?}] Fork tensor on {:?} to {:?}",
                std::thread::current().id(),
                device_here,
                device
            );
            let mut tensor = tensor.to_device(device).detach();
            println!(
                "[{:?}]-[COMPLETED] Fork tensor on {:?} to {:?}",
                std::thread::current().id(),
                device_here,
                device
            );

            if is_require_grad {
                tensor = tensor.require_grad();
            }

            tensor
        })
    }

    fn collect_devices(&self, mut devices: Vec<Device<B>>) -> Vec<Device<B>> {
        let device = self.val().device();

        if !devices.contains(&device) {
            devices.push(device)
        }

        devices
    }
}

impl<const D: usize, B: Backend> ModuleDisplayDefault for Param<Tensor<B, D>> {
    fn content(&self, content: Content) -> Option<Content> {
        let id = if content.display_settings.show_param_id() {
            format!(", id: {}", self.id)
        } else {
            "".to_string()
        };
        let string = format!(
            "ParamTensor {{rank: {D}, shape: {:?}, kind: float{id}}}",
            self.shape().dims
        );
        content.add_formatted(&string).optional()
    }
}
impl<const D: usize, B: Backend> ModuleDisplay for Param<Tensor<B, D>> {}

impl<const D: usize, B: Backend> Module<B> for Param<Tensor<B, D, Int>> {
    type Record = Param<Tensor<B, D, Int>>;

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        visitor.visit_int(self)
    }

    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        mapper.map_int(self)
    }

    fn into_record(self) -> Self::Record {
        self.transform_for_save()
    }

    fn load_record(self, record: Self::Record) -> Self {
        let (record_param_id, record_tensor, _) = record.consume();
        self.transform_for_load(record_tensor, record_param_id)
    }

    fn to_device(self, device: &Device<B>) -> Self {
        self.map(|tensor| tensor.to_device(device))
    }

    fn fork(self, device: &Device<B>) -> Self {
        self.to_device(device) // Don't support autodiff.
    }

    fn collect_devices(&self, mut devices: Vec<Device<B>>) -> Vec<Device<B>> {
        let device = self.val().device();

        if !devices.contains(&device) {
            devices.push(device)
        }

        devices
    }
}

impl<const D: usize, B: Backend> ModuleDisplayDefault for Param<Tensor<B, D, Int>> {
    fn content(&self, content: Content) -> Option<Content> {
        let id = if content.display_settings.show_param_id() {
            format!(", id: {}", self.id)
        } else {
            "".to_string()
        };
        let string = format!(
            "ParamTensor {{rank: {D}, shape: {:?}, kind: int{id}}}",
            self.shape().dims
        );
        content.add_formatted(&string).optional()
    }
}
impl<const D: usize, B: Backend> ModuleDisplay for Param<Tensor<B, D, Int>> {}

impl<const D: usize, B: Backend> Module<B> for Param<Tensor<B, D, Bool>> {
    type Record = Param<Tensor<B, D, Bool>>;

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        visitor.visit_bool(self)
    }

    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        mapper.map_bool(self)
    }

    fn into_record(self) -> Self::Record {
        self.transform_for_save()
    }

    fn load_record(self, record: Self::Record) -> Self {
        let (record_param_id, record_tensor, _) = record.consume();
        self.transform_for_load(record_tensor, record_param_id)
    }

    fn to_device(self, device: &Device<B>) -> Self {
        self.map(|tensor| tensor.to_device(device))
    }

    fn fork(self, device: &Device<B>) -> Self {
        self.to_device(device) // Don't support autodiff.
    }

    fn collect_devices(&self, mut devices: Vec<Device<B>>) -> Vec<Device<B>> {
        let device = self.val().device();

        if !devices.contains(&device) {
            devices.push(device)
        }

        devices
    }
}

impl<const D: usize, B: Backend> ModuleDisplayDefault for Param<Tensor<B, D, Bool>> {
    fn content(&self, content: Content) -> Option<Content> {
        let id = if content.display_settings.show_param_id() {
            format!(", id: {}", self.id)
        } else {
            "".to_string()
        };

        let string = format!(
            "ParamTensor {{rank: {D}, shape: {:?}, kind: bool{id}}}",
            self.shape().dims
        );
        content.add_formatted(&string).optional()
    }
}

impl<const D: usize, B: Backend> ModuleDisplay for Param<Tensor<B, D, Bool>> {}

impl<const D: usize, B: AutodiffBackend> AutodiffModule<B> for Param<Tensor<B, D>> {
    type InnerModule = Param<Tensor<B::InnerBackend, D>>;

    fn valid(&self) -> Self::InnerModule {
        Param::initialized(self.id, self.val().inner().set_require_grad(false))
    }
}

impl<const D: usize, B: AutodiffBackend> AutodiffModule<B> for Param<Tensor<B, D, Int>> {
    type InnerModule = Param<Tensor<B::InnerBackend, D, Int>>;

    fn valid(&self) -> Self::InnerModule {
        Param::initialized(self.id, self.val().inner())
    }
}

impl<const D: usize, B: AutodiffBackend> AutodiffModule<B> for Param<Tensor<B, D, Bool>> {
    type InnerModule = Param<Tensor<B::InnerBackend, D, Bool>>;

    fn valid(&self) -> Self::InnerModule {
        Param::initialized(self.id, self.val().inner())
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::{
        TestAutodiffBackend,
        module::Module,
        record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
    };

    #[test]
    fn test_load_record_setting() {
        let device = Default::default();
        let tensor = Tensor::<TestAutodiffBackend, 2>::ones([3, 3], &device).require_grad();

        let byte_recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let bytes = byte_recorder
            .record(
                Param::initialized(ParamId::new(), tensor.clone()).into_record(),
                (),
            )
            .unwrap();

        let no_grad_is_require_grad = Param::initialized(ParamId::new(), tensor.clone())
            .no_grad()
            .load_record(byte_recorder.load(bytes.clone(), &device).unwrap())
            .is_require_grad();

        let with_default_is_require_grad = Param::initialized(ParamId::new(), tensor)
            .load_record(byte_recorder.load(bytes, &device).unwrap())
            .is_require_grad();

        assert!(!no_grad_is_require_grad);
        assert!(with_default_is_require_grad);
    }
}
