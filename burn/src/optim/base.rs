use super::visitor::{GradientsLoader, GradientsRegister, ModuleTensorUpdater};
use crate::module::{LoadingError, Module, ParamId, State, StateNamed};
use crate::tensor::backend::{ADBackend, Backend};
use crate::tensor::{Data, Tensor};
use burn_tensor::backend::Gradients;

pub trait Optimizer: Send + Sync {
    type Backend: ADBackend;

    /// Update the tensor parameter using the given the gradients.
    fn update_tensor<const D: usize>(
        &mut self,
        id: &ParamId,
        tensor: &mut Tensor<Self::Backend, D>,
        grads: &<Self::Backend as ADBackend>::Gradients,
    );

    /// Update the parameters of the given module using the given the gradients.
    fn update_module<M>(&mut self, module: &mut M, grads: &<Self::Backend as ADBackend>::Gradients)
    where
        M: Module<Backend = Self::Backend>,
        Self: Sized,
    {
        let mut visitor = ModuleTensorUpdater::new(self, grads);
        module.visit_mut(&mut visitor);
    }

    /// Register the optimizer state for a given parameter.
    ///
    /// # Note
    ///
    /// This should only be called by generated code.
    fn register_param_state<const D: usize>(
        &self,
        _id: &ParamId,
        _state: &mut StateNamed<<Self::Backend as Backend>::Elem>,
    ) {
        // By default there is no state to register
    }

    /// Load the optimizer state for a given parameter.
    ///
    /// # Note
    ///
    /// This should only be called by generated code.
    fn load_param_state<const D: usize>(
        &mut self,
        _id: &ParamId,
        _state: &StateNamed<<Self::Backend as Backend>::Elem>,
        _device: &<Self::Backend as Backend>::Device,
    ) {
        // By default there is no state to load
    }

    /// Get the optimizer state for a given module.
    fn state<M: Module<Backend = Self::Backend>>(
        &self,
        module: &M,
    ) -> State<<Self::Backend as Backend>::Elem>
    where
        Self: Sized,
    {
        let mut state_named = StateNamed::new();
        let mut visitor = GradientsRegister::new(self, &mut state_named);

        module.visit(&mut visitor);
        State::StateNamed(state_named)
    }

    /// Load the optimizer state for a given module.
    fn load<M: Module<Backend = Self::Backend>>(
        &mut self,
        module: &M,
        state: &State<<Self::Backend as Backend>::Elem>,
    ) -> Result<(), LoadingError>
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

pub(super) fn register_state_gradients<const D: usize, B: ADBackend, F: Fn(&str) -> String>(
    id: &ParamId,
    state: &mut StateNamed<B::Elem>,
    grads: &B::Gradients,
    id_to_key: F,
) {
    let id = id.to_string();

    if let Some(velocity) = grads.get::<D>(&id) {
        let data = State::Data(velocity.to_data().serialize());
        state.register_state(id_to_key(&id).as_str(), data);
    };
}

pub(super) fn load_state_gradients<const D: usize, B: ADBackend, F: Fn(&str) -> String>(
    id: &ParamId,
    state: &StateNamed<B::Elem>,
    grads: &mut B::Gradients,
    id_to_key: F,
    device: &B::Device,
) {
    let id = id.to_string();

    if let Some(State::Data(data)) = state.get(id_to_key(&id).as_str()) {
        let velocity = Tensor::<B::InnerBackend, D>::from_data_device(Data::from(data), *device);
        grads.register(id, velocity);
    };
}
