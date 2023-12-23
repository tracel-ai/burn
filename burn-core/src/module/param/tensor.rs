use super::{Param, ParamId};
use crate::module::{AutodiffModule, Module, ModuleMapper, ModuleVisitor};
use crate::tensor::{
    backend::{AutodiffBackend, Backend},
    Tensor,
};
use alloc::vec::Vec;
use burn_tensor::{Bool, Int};

impl<B: Backend, const D: usize> From<Tensor<B, D>> for Param<Tensor<B, D>> {
    fn from(value: Tensor<B, D>) -> Self {
        // When creating a parameter from a float tensor, we automatically mark it as requiring
        // gradients, so that it can be updated by an optimizer.
        Param::new(ParamId::new(), value.require_grad())
    }
}

impl<B: Backend, const D: usize> From<Tensor<B, D, Int>> for Param<Tensor<B, D, Int>> {
    fn from(value: Tensor<B, D, Int>) -> Self {
        Param::new(ParamId::new(), value)
    }
}

impl<B: Backend, const D: usize> From<Tensor<B, D, Bool>> for Param<Tensor<B, D, Bool>> {
    fn from(value: Tensor<B, D, Bool>) -> Self {
        Param::new(ParamId::new(), value)
    }
}

impl<const D: usize, B: Backend> Module<B> for Param<Tensor<B, D>> {
    type Record = Param<Tensor<B, D>>;

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        visitor.visit_float(&self.id, &self.value)
    }

    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        let value = mapper.map_float(&self.id, self.value);
        Self::new(self.id, value)
    }

    fn into_record(self) -> Self::Record {
        self
    }

    fn load_record(self, record: Self::Record) -> Self {
        let mut tensor = record.value.detach();
        let device = self.device();

        // Make sure we load the record into the same module device.
        if tensor.device() != device {
            tensor = tensor.to_device(&device).detach();
        }

        // Make sure we load the record with the same autodiff setting.
        tensor = tensor.set_require_grad(self.is_require_grad());

        Self::new(record.id, tensor)
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
        let device = self.device();

        if !devices.contains(&device) {
            devices.push(device)
        }

        devices
    }
}

impl<const D: usize, B: Backend> Module<B> for Param<Tensor<B, D, Int>> {
    type Record = Param<Tensor<B, D, Int>>;

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        visitor.visit_int(&self.id, &self.value)
    }

    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        let value = mapper.map_int(&self.id, self.value);
        Self::new(self.id, value)
    }

    fn into_record(self) -> Self::Record {
        self
    }

    fn load_record(self, record: Self::Record) -> Self {
        let mut tensor = record.value;
        let device = self.device();

        // Make sure we load the record into the same module device.
        if tensor.device() != device {
            tensor = tensor.to_device(&device);
        }

        Self::new(record.id, tensor)
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
        let device = self.device();

        if !devices.contains(&device) {
            devices.push(device)
        }

        devices
    }
}

impl<const D: usize, B: Backend> Module<B> for Param<Tensor<B, D, Bool>> {
    type Record = Param<Tensor<B, D, Bool>>;

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        visitor.visit_bool(&self.id, &self.value)
    }

    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        let value = mapper.map_bool(&self.id, self.value);
        Self::new(self.id, value)
    }

    fn into_record(self) -> Self::Record {
        self
    }

    fn load_record(self, record: Self::Record) -> Self {
        let mut tensor = record.value;
        let device = self.device();

        // Make sure we load the record into the same module device.
        if tensor.device() != device {
            tensor = tensor.to_device(&device);
        }

        Self::new(record.id, tensor)
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
        let device = self.device();

        if !devices.contains(&device) {
            devices.push(device)
        }

        devices
    }
}

impl<const D: usize, B: AutodiffBackend> AutodiffModule<B> for Param<Tensor<B, D>> {
    type InnerModule = Param<Tensor<B::InnerBackend, D>>;

    fn valid(&self) -> Self::InnerModule {
        Param::new(
            self.id.clone(),
            self.value.clone().inner().set_require_grad(false),
        )
    }
}

impl<const D: usize, B: AutodiffBackend> AutodiffModule<B> for Param<Tensor<B, D, Int>> {
    type InnerModule = Param<Tensor<B::InnerBackend, D, Int>>;

    fn valid(&self) -> Self::InnerModule {
        Param::new(self.id.clone(), self.value.clone().inner())
    }
}

impl<const D: usize, B: AutodiffBackend> AutodiffModule<B> for Param<Tensor<B, D, Bool>> {
    type InnerModule = Param<Tensor<B::InnerBackend, D, Bool>>;

    fn valid(&self) -> Self::InnerModule {
        Param::new(self.id.clone(), self.value.clone().inner())
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::{
        module::Module,
        nn::LinearConfig,
        record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
        TestAutodiffBackend,
    };

    #[test]
    fn test_load_record_setting() {
        let tensor = Tensor::<TestAutodiffBackend, 2>::ones_devauto([3, 3]);

        let byte_recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let bytes = byte_recorder
            .record(Param::from(tensor.clone()).into_record(), ())
            .unwrap();

        let no_grad_is_require_grad = Param::from(tensor.clone())
            .no_grad()
            .load_record(byte_recorder.load(bytes.clone()).unwrap())
            .value
            .is_require_grad();

        let with_default_is_require_grad = Param::from(tensor)
            .load_record(byte_recorder.load(bytes).unwrap())
            .value
            .is_require_grad();

        assert!(!no_grad_is_require_grad);
        assert!(with_default_is_require_grad);
    }

    #[test]
    fn test_init_with_record_setting() {
        let config = LinearConfig::new(32, 32);
        let device = Default::default();
        let module_init = config.init::<TestAutodiffBackend>(&device);

        let record = module_init.clone().into_record();
        let module_init_with = config.init_with::<TestAutodiffBackend>(record);

        assert_eq!(
            module_init.weight.is_require_grad(),
            module_init_with.weight.is_require_grad()
        );
    }
}
