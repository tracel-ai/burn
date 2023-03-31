use super::mapper::ModuleTensorUpdater;
use super::visitor::{GradientsLoader, GradientsRegister};
use super::GradientsParams;

use crate::module::{ADModule, LoadingError, ParamId, State, StateNamed};
use crate::tensor::backend::ADBackend;
use crate::tensor::{Data, Tensor};

pub trait Optimizer<M, B>: Send + Sync
where
    M: ADModule<B>,
    B: ADBackend,
{
    /// Update the tensor parameter using the given the gradients.
    fn update_tensor<const D: usize>(
        &mut self,
        id: &ParamId,
        tensor: Tensor<B, D>,
        grad: Tensor<B::InnerBackend, D>,
    ) -> Tensor<B, D>;

    /// Update the parameters of the given module using the given the gradients.
    fn update_module(&mut self, module: M, grads: GradientsParams) -> M
    where
        Self: Sized,
    {
        let mut mapper = ModuleTensorUpdater::new(self, grads);
        module.map(&mut mapper)
    }

    /// Register the optimizer state for a given parameter.
    ///
    /// # Notes
    ///
    /// This should only be called by generated code.
    fn register_param_state<const D: usize>(
        &self,
        _id: &ParamId,
        _state: &mut StateNamed<B::FloatElem>,
    ) {
        // By default there is no state to register
    }

    /// Load the optimizer state for a given parameter.
    ///
    /// # Notes
    ///
    /// This should only be called by generated code.
    fn load_param_state<const D: usize>(
        &mut self,
        _id: &ParamId,
        _state: &StateNamed<B::FloatElem>,
        _device: &B::Device,
    ) {
        // By default there is no state to load
    }

    /// Get the optimizer state for a given module.
    fn state(&self, module: &M) -> State<B::FloatElem>
    where
        Self: Sized,
    {
        let mut state_named = StateNamed::new();
        let mut visitor = GradientsRegister::new(self, &mut state_named);

        module.visit(&mut visitor);
        State::StateNamed(state_named)
    }

    /// Load the optimizer state for a given module.
    fn load(&mut self, module: &M, state: &State<B::FloatElem>) -> Result<(), LoadingError>
    where
        Self: Sized,
    {
        let state_named = match state {
            State::StateNamed(state) => state,
            _ => {
                return Err(LoadingError::new(
                    "Can't load state wrapper to fetch id and data".to_string(),
                ))
            }
        };

        let mut visitor = GradientsLoader::new(self, state_named);
        module.visit(&mut visitor);

        Ok(())
    }
}

pub(super) fn register_state_gradients<const D: usize, B: ADBackend, F: Fn(&ParamId) -> String>(
    id: &ParamId,
    state: &mut StateNamed<B::FloatElem>,
    grads: &GradientsParams,
    id_to_key: F,
) {
    if let Some(grad) = grads.get::<B::InnerBackend, D>(id) {
        let data = State::Data(grad.into_data().serialize());
        state.register_state(id_to_key(id).as_str(), data);
    };
}

pub(super) fn load_state_gradients<const D: usize, B: ADBackend, F: Fn(&ParamId) -> String>(
    id: &ParamId,
    state: &StateNamed<B::FloatElem>,
    grads: &mut GradientsParams,
    id_to_key: F,
    device: &B::Device,
) {
    if let Some(State::Data(data)) = state.get(id_to_key(id).as_str()) {
        let tensor = Tensor::<B::InnerBackend, D>::from_data_device(Data::from(data), device);
        grads.register::<B::InnerBackend, D>(id.clone(), tensor);
    };
}
