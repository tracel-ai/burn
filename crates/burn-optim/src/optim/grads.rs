use burn_core as burn;

use burn::{
    Tensor,
    tensor::{Device, Gradients, container::TensorContainer},
};

use burn::module::{AutodiffModule, ParamId};

use super::visitor::{GradientsParamsChangeDevice, GradientsParamsConverter};

/// Data type that contains gradients for parameters.
#[derive(Default, Debug)]
pub struct GradientsParams {
    container: TensorContainer<ParamId>,
}

impl GradientsParams {
    /// Creates a new [GradientsParams](GradientsParams).
    pub fn new() -> Self {
        Self::default()
    }

    /// Extract each tensor gradients for the given [module](AutodiffModule).
    ///
    /// Note: This consumes the gradients. See ['from_module'] to extract gradients only for
    ///  a specific module.
    pub fn from_grads<M: AutodiffModule>(grads: Gradients, module: &M) -> Self {
        let mut grads = grads;
        Self::from_module(&mut grads, module)
    }

    /// Extract each tensor gradients for the given [module](AutodiffModule).
    pub fn from_module<M: AutodiffModule>(grads: &mut Gradients, module: &M) -> Self {
        let mut grads_params = GradientsParams::new();
        let mut visitor = GradientsParamsConverter::<M>::new(grads, &mut grads_params, None);
        module.visit(&mut visitor);
        grads_params
    }

    /// Extract tensor gradients for the given [module](AutodiffModule) and given parameters.
    pub fn from_params<M: AutodiffModule>(
        grads: &mut Gradients,
        module: &M,
        params: &[ParamId],
    ) -> Self {
        let mut grads_params = GradientsParams::new();
        let mut visitor =
            GradientsParamsConverter::<M>::new(grads, &mut grads_params, Some(params.to_vec()));
        module.visit(&mut visitor);
        grads_params
    }

    /// Get the gradients for the given [parameter id](ParamId).
    ///
    /// # Notes
    ///
    /// You should use [remove](GradientsParams::remove) if you want to get the gradients
    /// only one time.
    pub fn get<const D: usize>(&self, id: ParamId) -> Option<Tensor<D>> {
        self.container.get(&id).map(Tensor::from_primitive)
    }

    /// Remove the gradients for the given [parameter id](ParamId).
    pub fn remove<const D: usize>(&mut self, id: ParamId) -> Option<Tensor<D>> {
        self.container.remove(&id).map(Tensor::from_primitive)
    }

    /// Register a gradients tensor for the given [parameter id](ParamId).
    ///
    /// # Notes
    ///
    /// If a tensor is already registered for the given [parameter id](ParamId), it will be replaced.
    pub fn register<const D: usize>(&mut self, id: ParamId, value: Tensor<D>) {
        // TODO: always call value.inner() to make sure?
        self.container.register(id, value.into_primitive())
    }

    /// The number of gradients tensors registered.
    pub fn len(&self) -> usize {
        self.container.len()
    }

    /// If any tensor is contained.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Change the device of each tensor gradients registered for the given [module](AutodiffModule).
    pub fn to_device<M: AutodiffModule>(mut self, device: &Device, module: &M) -> Self {
        let mut visitor = GradientsParamsChangeDevice::<M>::new(device, &mut self);
        module.visit(&mut visitor);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::module::{Module, list_param_ids};
    use burn::tensor::Distribution;
    use burn_nn::{Linear, LinearConfig};

    #[test]
    fn test_convert_grads() {
        let device = Device::default().autodiff();
        let layer_1 = layer(&device);
        let mut layer_2 = layer_1.clone();
        layer_2 = layer_2.fork(&device);
        let loss_1 = layer_1.forward(random_tensor(&device));
        let loss_2 = layer_2.forward(random_tensor(&device));
        let grads_1 = GradientsParams::from_grads(loss_1.backward(), &layer_1);
        let grads_2 = GradientsParams::from_grads(loss_2.backward(), &layer_2);

        let param_ids_1 = list_param_ids(&layer_1);
        let param_ids_2 = list_param_ids(&layer_2);

        assert_eq!(param_ids_1, param_ids_2);
        assert_eq!(grads_1.len(), param_ids_1.len());
        assert_eq!(grads_2.len(), param_ids_2.len());
    }

    fn layer(device: &Device) -> Linear {
        LinearConfig::new(20, 20).init(device)
    }

    fn random_tensor(device: &Device) -> Tensor<2> {
        Tensor::<2>::random([2, 20], Distribution::Default, device)
    }
}
