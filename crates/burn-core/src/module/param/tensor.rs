use super::{Param, ParamId, Parameter};
use crate::module::{
    AutodiffModule, Content, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper,
    ModuleVisitor,
};
use crate::tensor::{
    backend::{AutodiffBackend, Backend},
    Tensor,
};
use alloc::{format, string::ToString, vec::Vec};
use burn_tensor::{Bool, Data, Float, Int};

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

    /// Create a new parameter from data.
    pub fn from_data<T>(data: T, device: &B::Device) -> Self
    where
        T: Into<Data<B::FloatElem, D>>,
    {
        // When creating a parameter from a float tensor, we automatically mark it as requiring
        // gradients, so that it can be updated by an optimizer.
        let value = Tensor::from_data(data, device);
        Param::initialized(ParamId::new(), value.require_grad())
    }
}

impl<const D: usize, B: Backend> Module<B> for Param<Tensor<B, D>> {
    type Record = Param<Tensor<B, D>>;

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        visitor.visit_float(&self.id, &self.val())
    }

    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        let (id, tensor) = self.consume();
        let value = mapper.map_float(&id, tensor);

        Self::initialized(id, value)
    }

    fn into_record(self) -> Self::Record {
        self
    }

    fn load_record(self, record: Self::Record) -> Self {
        let (new_id, mut new_value) = record.consume();

        let expected_device = self.lazy_device();
        let expected_require_grad = self.lazy_is_require_grad();

        // Make sure we load the record into the same module device.
        if new_value.device() != expected_device {
            new_value = new_value.to_device(&expected_device).detach();
        }

        // Make sure we load the record with the same autodiff setting.
        new_value = new_value.set_require_grad(expected_require_grad);

        Self::initialized(new_id, new_value)
    }

    fn to_device(self, device: &<B as Backend>::Device) -> Self {
        self.map(|tensor| tensor.to_device(device))
    }

    fn fork(self, device: &<B as Backend>::Device) -> Self {
        self.map(|tensor| {
            let is_require_grad = tensor.is_require_grad();
            let mut tensor = tensor.to_device(device).detach();

            if is_require_grad {
                tensor = tensor.require_grad();
            }

            tensor
        })
    }

    fn collect_devices(
        &self,
        mut devices: Vec<<B as Backend>::Device>,
    ) -> Vec<<B as Backend>::Device> {
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
        visitor.visit_int(&self.id, &self.val())
    }

    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        let value = mapper.map_int(&self.id, self.val());
        Self::initialized(self.id, value)
    }

    fn into_record(self) -> Self::Record {
        self
    }

    fn load_record(self, record: Self::Record) -> Self {
        let (new_id, mut new_value) = record.consume();

        let expected_device = self.lazy_device();

        // Make sure we load the record into the same module device.
        if new_value.device() != expected_device {
            new_value = new_value.to_device(&expected_device);
        }

        Self::initialized(new_id, new_value)
    }

    fn to_device(self, device: &<B as Backend>::Device) -> Self {
        self.map(|tensor| tensor.to_device(device))
    }

    fn fork(self, device: &<B as Backend>::Device) -> Self {
        self.to_device(device) // Don't support autodiff.
    }

    fn collect_devices(
        &self,
        mut devices: Vec<<B as Backend>::Device>,
    ) -> Vec<<B as Backend>::Device> {
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
        visitor.visit_bool(&self.id, &self.val())
    }

    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        let value = mapper.map_bool(&self.id, self.val());
        Self::initialized(self.id, value)
    }

    fn into_record(self) -> Self::Record {
        self
    }

    fn load_record(self, record: Self::Record) -> Self {
        let (new_id, mut new_value) = record.consume();

        let expected_device = self.lazy_device();

        // Make sure we load the record into the same module device.
        if new_value.device() != expected_device {
            new_value = new_value.to_device(&expected_device);
        }

        Self::initialized(new_id, new_value)
    }

    fn to_device(self, device: &<B as Backend>::Device) -> Self {
        self.map(|tensor| tensor.to_device(device))
    }

    fn fork(self, device: &<B as Backend>::Device) -> Self {
        self.to_device(device) // Don't support autodiff.
    }

    fn collect_devices(
        &self,
        mut devices: Vec<<B as Backend>::Device>,
    ) -> Vec<<B as Backend>::Device> {
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
        Param::initialized(self.id.clone(), self.val().inner().set_require_grad(false))
    }
}

impl<const D: usize, B: AutodiffBackend> AutodiffModule<B> for Param<Tensor<B, D, Int>> {
    type InnerModule = Param<Tensor<B::InnerBackend, D, Int>>;

    fn valid(&self) -> Self::InnerModule {
        Param::initialized(self.id.clone(), self.val().inner())
    }
}

impl<const D: usize, B: AutodiffBackend> AutodiffModule<B> for Param<Tensor<B, D, Bool>> {
    type InnerModule = Param<Tensor<B::InnerBackend, D, Bool>>;

    fn valid(&self) -> Self::InnerModule {
        Param::initialized(self.id.clone(), self.val().inner())
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::{
        module::Module,
        record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
        TestAutodiffBackend,
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
