use crate::{OptimizerStateMapper, OptimizerStateStorage};
use burn_core::tensor::Tensor;

#[derive(Clone)]
pub(super) struct StorageOptimizerStateMapper {
    storage: OptimizerStateStorage,
}
impl StorageOptimizerStateMapper {
    pub(super) fn new(storage: OptimizerStateStorage) -> Self {
        Self { storage }
    }
}
impl OptimizerStateMapper for StorageOptimizerStateMapper {
    fn map<const D: usize>(&mut self, tensor: Tensor<D>, _name: &str) -> Tensor<D> {
        match self.storage {
            OptimizerStateStorage::Float(dtype) => tensor.cast(dtype),
            OptimizerStateStorage::Quantized(scheme) => tensor.quantize_dynamic(&scheme),
        }
    }
}
