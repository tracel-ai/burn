use burn_tensor::container::TensorContainer;

use super::mapper::ModuleTensorUpdater;
use super::visitor::{
    GradientsLoader, GradientsParamsChangeDevice, GradientsParamsConverter, GradientsRegister,
};

use crate::module::{ADModule, LoadingError, Module, ParamId, State, StateNamed};
use crate::tensor::backend::{ADBackend, Backend};
use crate::tensor::{Data, Tensor};

/// Data type that contains gradients for a given backend.
pub type GradientsParams = TensorContainer<ParamId>;

pub trait Optimizer: Send + Sync {
    type Backend: ADBackend;

    /// Update the tensor parameter using the given the gradients.
    fn update_tensor<const D: usize>(
        &mut self,
        id: &ParamId,
        tensor: Tensor<Self::Backend, D>,
        grad: Tensor<<Self::Backend as ADBackend>::InnerBackend, D>,
    ) -> Tensor<Self::Backend, D>;

    /// Update the parameters of the given module using the given the gradients.
    fn update_module<M>(&mut self, module: M, grads: GradientsParams) -> M
    where
        M: ADModule<ADBackend = Self::Backend>,
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
        _state: &mut StateNamed<<Self::Backend as Backend>::FloatElem>,
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
        _state: &StateNamed<<Self::Backend as Backend>::FloatElem>,
        _device: &<Self::Backend as Backend>::Device,
    ) {
        // By default there is no state to load
    }

    /// Get the optimizer state for a given module.
    fn state<M: Module<Backend = Self::Backend>>(
        &self,
        module: &M,
    ) -> State<<Self::Backend as Backend>::FloatElem>
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
        state: &State<<Self::Backend as Backend>::FloatElem>,
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

pub(super) fn register_state_gradients<const D: usize, B: ADBackend, F: Fn(&ParamId) -> String>(
    id: &ParamId,
    state: &mut StateNamed<B::FloatElem>,
    grads: &GradientsParams,
    id_to_key: F,
) {
    if let Some(grad) = grads.get::<B::InnerBackend, D>(id) {
        let data = State::Data(grad.to_data().serialize());
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

/// Update the device of each tensor gradients.
pub fn to_device_grads<M: ADModule>(
    grads: &mut GradientsParams,
    device: <M::Backend as Backend>::Device,
    module: &M,
) {
    let mut visitor = GradientsParamsChangeDevice::new(device, grads);
    module.visit(&mut visitor);
}

/// Convert the gradients returned by the ADBackend into a tensor container that contains
/// gradients corresponding to the given module.
pub fn convert_grads<M: ADModule>(
    grads: <M::ADBackend as ADBackend>::Gradients,
    module: &M,
) -> GradientsParams {
    let mut grads_params = TensorContainer::new();
    let mut visitor = GradientsParamsConverter::new(grads, &mut grads_params);
    module.visit(&mut visitor);

    grads_params
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        module::{list_param_ids, Module},
        nn::{Linear, LinearConfig},
        TestADBackend,
    };
    use burn_tensor::{backend::Backend, Distribution};

    #[test]
    fn test_convert_grads() {
        let layer_1 = layer();
        let mut layer_2 = layer_1.clone();
        layer_2 = layer_2
            .to_device(&<TestADBackend as Backend>::Device::default())
            .detach();
        let loss_1 = layer_1.forward(random_tensor());
        let loss_2 = layer_2.forward(random_tensor());
        let grads_1 = loss_1.backward();
        let grads_2 = loss_2.backward();

        convert_grads(grads_1, &layer_1);
        convert_grads(grads_2, &layer_2);

        let param_ids_1 = list_param_ids(&layer_1);
        let params_ids_2 = list_param_ids(&layer_2);

        assert_eq!(param_ids_1, params_ids_2);
    }

    fn layer() -> Linear<TestADBackend> {
        Linear::<TestADBackend>::new(&LinearConfig {
            d_input: 20,
            d_output: 20,
            bias: true,
        })
    }

    fn random_tensor() -> Tensor<TestADBackend, 2> {
        Tensor::<TestADBackend, 2>::random([2, 20], Distribution::Standard)
    }
}
