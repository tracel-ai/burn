use crate::module::{ParamId, StateNamed};
use crate::tensor::backend::{ADBackend, Backend};
use crate::tensor::{Gradients, Tensor};

pub trait Optimizer: Send + Sync {
    type Backend: ADBackend;

    fn update<const D: usize>(
        &mut self,
        id: &ParamId,
        tensor: &mut Tensor<Self::Backend, D>,
        grads: &Gradients,
    );

    fn register_state<const D: usize>(
        &self,
        id: &ParamId,
        state: &mut StateNamed<<Self::Backend as Backend>::Elem>,
    );

    fn load_state<const D: usize>(
        &mut self,
        id: &ParamId,
        state: &StateNamed<<Self::Backend as Backend>::Elem>,
        device: &<Self::Backend as Backend>::Device,
    );
}
