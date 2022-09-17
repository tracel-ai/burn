use crate::module::ParamId;
use crate::tensor::backend::ADBackend;
use crate::tensor::{Gradients, Tensor};

pub trait Optimizer: Send + Sync {
    type Backend: ADBackend;

    fn update<const D: usize>(
        &mut self,
        id: &ParamId,
        tensor: &mut Tensor<Self::Backend, D>,
        grads: &Gradients,
    );
}
