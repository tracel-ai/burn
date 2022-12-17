use crate::module::{
    LoadingError, Module, ModuleVisitor, ModuleVisitorMut, ParamId, State, StateNamed,
};
use crate::tensor::backend::{ADBackend, Backend};
use crate::tensor::{Data, Tensor};
use burn_tensor::backend::Gradients;

pub trait Optimizer: Send + Sync {
    type Backend: ADBackend;

    fn update_tensor<const D: usize>(
        &mut self,
        id: &ParamId,
        tensor: &mut Tensor<Self::Backend, D>,
        grads: &<Self::Backend as ADBackend>::Gradients,
    );

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

    fn update_module<M: Module>(
        &mut self,
        module: &mut M,
        grads: &<Self::Backend as ADBackend>::Gradients,
    ) where
        M: Module<Backend = Self::Backend>,
        Self: Sized,
    {
        let mut visitor = ModuleOptimizer {
            optimizer: self,
            grads,
        };
        module.visit_mut(&mut visitor);
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
        let mut visitor = GradientsRegistering {
            optimizer: self,
            state: &mut state_named,
        };
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

        let mut visitor = GradientsLoading {
            optimizer: self,
            state: state_named,
        };
        module.visit(&mut visitor);

        Ok(())
    }
}

struct GradientsRegistering<'a, B: ADBackend, O> {
    optimizer: &'a O,
    state: &'a mut StateNamed<B::Elem>,
}

impl<'a, B: ADBackend, O: Optimizer<Backend = B>> ModuleVisitor<B>
    for GradientsRegistering<'a, B, O>
{
    fn visit<const D: usize>(&mut self, id: &ParamId, _tensor: &Tensor<B, D>) {
        self.optimizer
            .register_param_state::<D>(id, &mut self.state)
    }
}

struct GradientsLoading<'a, B: ADBackend, O> {
    optimizer: &'a mut O,
    state: &'a StateNamed<B::Elem>,
}

struct ModuleOptimizer<'a, B: ADBackend, O> {
    optimizer: &'a mut O,
    grads: &'a B::Gradients,
}

impl<'a, B: ADBackend, O: Optimizer<Backend = B>> ModuleVisitorMut<B>
    for ModuleOptimizer<'a, B, O>
{
    fn visit_mut<const D: usize>(&mut self, id: &ParamId, tensor: &mut Tensor<B, D>) {
        self.optimizer.update_tensor(id, tensor, &self.grads);
    }
}

impl<'a, B: ADBackend, O: Optimizer<Backend = B>> ModuleVisitor<B> for GradientsLoading<'a, B, O> {
    fn visit<const D: usize>(&mut self, id: &ParamId, tensor: &Tensor<B, D>) {
        self.optimizer
            .load_param_state::<D>(id, &mut self.state, &tensor.device())
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
