use burn_tensor::{
    backend::{ADBackend, Backend},
    container::TensorContainer,
    Tensor,
};

use crate::module::{ADModule, ParamId};

use super::visitor::{GradientsParamsChangeDevice, GradientsParamsConverter};

/// Data type that contains gradients for parameters.
#[derive(Default)]
pub struct GradientsParams {
    container: TensorContainer<ParamId>,
}

impl GradientsParams {
    /// Creates a new [GradientsParams](GradientsParams).
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the gradients for the given [parameter id](ParamId).
    ///
    /// # Notes
    ///
    /// You should use [remove](GradientsParams::remove) if you want to get the gradients
    /// only one time.
    pub fn get<B, const D: usize>(&self, id: &ParamId) -> Option<Tensor<B, D>>
    where
        B: Backend,
    {
        self.container.get(id)
    }

    /// Remove the gradients for the given [parameter id](ParamId).
    pub fn remove<B, const D: usize>(&mut self, id: &ParamId) -> Option<Tensor<B, D>>
    where
        B: Backend,
    {
        self.container.remove(id)
    }

    /// Register a gradients tensor for the given [parameter id](ParamId).
    ///
    /// # Notes
    ///
    /// If a tensor is already registered for the given [parameter id](ParamId), it will be replaced.
    pub fn register<B, const D: usize>(&mut self, id: ParamId, value: Tensor<B, D>)
    where
        B: Backend,
    {
        self.container.register(id, value)
    }

    /// The number of gradients tensors registered.
    pub fn len(&self) -> usize {
        self.container.len()
    }

    /// If any tensor is contained.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Change the device of each tensor gradients registered for the given [module](ADModule).
    pub fn to_device<B: ADBackend, M: ADModule<B>>(
        mut self,
        device: &B::Device,
        module: &M,
    ) -> Self {
        let mut visitor = GradientsParamsChangeDevice::<M, B>::new(device, &mut self);
        module.visit(&mut visitor);
        self
    }

    /// Extract each tensor gradients for the given [module](ADModule).
    pub fn from_grads<B: ADBackend, M: ADModule<B>>(grads: B::Gradients, module: &M) -> Self {
        let mut grads_params = GradientsParams::new();
        let mut visitor = GradientsParamsConverter::<M, B>::new(grads, &mut grads_params);

        module.visit(&mut visitor);
        grads_params
    }
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
        layer_2 = layer_2.fork(&<TestADBackend as Backend>::Device::default());
        let loss_1 = layer_1.forward(random_tensor());
        let loss_2 = layer_2.forward(random_tensor());
        let grads_1 = GradientsParams::from_grads(loss_1.backward(), &layer_1);
        let grads_2 = GradientsParams::from_grads(loss_2.backward(), &layer_2);

        let param_ids_1 = list_param_ids(&layer_1);
        let param_ids_2 = list_param_ids(&layer_2);

        assert_eq!(param_ids_1, param_ids_2);
        assert_eq!(grads_1.len(), param_ids_1.len());
        assert_eq!(grads_2.len(), param_ids_2.len());
    }

    fn layer() -> Linear<TestADBackend> {
        LinearConfig::new(20, 20).with_bias(true).init()
    }

    fn random_tensor() -> Tensor<TestADBackend, 2> {
        Tensor::<TestADBackend, 2>::random([2, 20], Distribution::Default)
    }
}
