use super::{Param, ParamId};
use crate::module::{ADModule, Module, ModuleMapper, ModuleVisitor};
use crate::tensor::{
    backend::{ADBackend, Backend},
    Tensor,
};

impl<B: Backend, const D: usize> From<Tensor<B, D>> for Param<Tensor<B, D>> {
    fn from(value: Tensor<B, D>) -> Self {
        Param::new(ParamId::new(), value.require_grad())
    }
}

impl<const D: usize, B: Backend> Module<B> for Param<Tensor<B, D>> {
    type Record = Param<Tensor<B, D>>;

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        visitor.visit(&self.id, &self.value)
    }

    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        let value = mapper.map(&self.id, self.value);
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
}

impl<const D: usize, B: ADBackend> ADModule<B> for Param<Tensor<B, D>> {
    type InnerModule = Param<Tensor<B::InnerBackend, D>>;

    fn valid(&self) -> Self::InnerModule {
        Param::new(
            self.id.clone(),
            self.value.clone().inner().set_require_grad(false),
        )
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::{
        module::Module,
        nn::LinearConfig,
        record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
        TestADBackend,
    };

    #[test]
    fn test_load_record_setting() {
        let tensor = Tensor::<TestADBackend, 2>::ones([3, 3]);

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
        let module_init = config.init::<TestADBackend>();

        let record = module_init.clone().into_record();
        let module_init_with = config.init_with::<TestADBackend>(record);

        assert_eq!(
            module_init.weight.is_require_grad(),
            module_init_with.weight.is_require_grad()
        );
    }
}
