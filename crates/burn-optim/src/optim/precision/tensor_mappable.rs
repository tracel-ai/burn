use crate::OptimizerStateMapper;
use burn_core::tensor::Tensor;

pub(crate) trait TensorMappable {
    fn map<M: OptimizerStateMapper>(self, mapper: &mut M, name: &str) -> Self;
}
impl<const D: usize> TensorMappable for Tensor<D> {
    fn map<M: OptimizerStateMapper>(self, mapper: &mut M, name: &str) -> Self {
        mapper.map(self, name)
    }
}
impl<T: TensorMappable> TensorMappable for Option<T> {
    fn map<M: OptimizerStateMapper>(self, mapper: &mut M, name: &str) -> Self {
        self.map(|tensor| tensor.map(mapper, name))
    }
}
