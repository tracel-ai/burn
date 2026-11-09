use burn_core as burn;

use burn::{
    Tensor,
    tensor::{Device, FloatDType, Gradients, container::TensorContainer},
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use burn::module::{Module, ParamId};

use super::visitor::{
    GradientsParamsChangeDevice, GradientsParamsConverter, GradientsParamsScale,
    GradientsParamsSquaredNorms,
};

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

    /// Extract each tensor gradients for the given [module](Module).
    ///
    /// Note: This consumes the gradients. See ['from_module'] to extract gradients only for
    ///  a specific module.
    pub fn from_grads<M: Module>(grads: Gradients, module: &M) -> Self {
        let mut grads = grads;
        Self::from_module(&mut grads, module)
    }

    /// Extract each tensor gradients for the given [module](Module).
    pub fn from_module<M: Module>(grads: &mut Gradients, module: &M) -> Self {
        let mut grads_params = GradientsParams::new();
        let mut visitor = GradientsParamsConverter::<M>::new(grads, &mut grads_params, None);
        module.visit(&mut visitor);
        grads_params
    }

    /// Extract tensor gradients for the given [module](Module) and given parameters.
    pub fn from_params<M: Module>(grads: &mut Gradients, module: &M, params: &[ParamId]) -> Self {
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
        self.container.get(&id)
    }

    /// Remove the gradients for the given [parameter id](ParamId).
    pub fn remove<const D: usize>(&mut self, id: ParamId) -> Option<Tensor<D>> {
        self.container.remove(&id)
    }

    /// Register a gradients tensor for the given [parameter id](ParamId).
    ///
    /// # Notes
    ///
    /// If a tensor is already registered for the given [parameter id](ParamId), it will be replaced.
    pub fn register<const D: usize>(&mut self, id: ParamId, value: Tensor<D>) {
        // TODO: always call value.inner() to make sure?
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

    /// Scales every gradient of `module` by one shared factor so that their combined L2 norm is at
    /// most `max_norm`, and returns that norm before clipping. This matches PyTorch's
    /// `clip_grad_norm_`; call it before [`step`](crate::optim::ModuleOptimizer::step).
    ///
    /// [`GradientClipping::Norm`](crate::grad_clipping::GradientClipping::Norm) instead clips each
    /// parameter's gradient by its own norm, so a model with many parameters can pass a much larger
    /// update overall. Returns `None` when no gradient is registered for `module`.
    pub fn clip_norm<M: Module>(&mut self, module: &M, max_norm: f32) -> Option<Tensor<1>> {
        let mut squares = Vec::new();
        module.visit(&mut GradientsParamsSquaredNorms::<M>::new(
            self,
            &mut squares,
        ));
        let device = squares.first()?.device();
        let squares: Vec<_> = squares.into_iter().map(|s| s.to_device(&device)).collect();
        let norm = Tensor::cat(squares, 0).sum().sqrt();

        let min_positive = FloatDType::F32.finfo().min_positive;
        let scale = (norm.clone().add_scalar(min_positive).recip() * max_norm).clamp_max(1.0);
        module.visit(&mut GradientsParamsScale::<M>::new(self, &scale));
        Some(norm)
    }

    /// Change the device of each tensor gradients registered for the given [module](Module).
    pub fn to_device<M: Module>(mut self, device: &Device, module: &M) -> Self {
        let mut visitor = GradientsParamsChangeDevice::<M>::new(device, &mut self);
        module.visit(&mut visitor);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::module::{Module, ParamGroup};
    use burn::tensor::{Distribution, TensorData, Tolerance};
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

        let group = ParamGroup::ids_from_module(layer_1.clone());
        let bias_2 = layer_2.bias.as_ref().unwrap();

        assert!(group.matches(&layer_2.weight.id, None));
        assert!(group.matches(&bias_2.id, None));
        assert_eq!(grads_1.len(), 2);
        assert_eq!(grads_2.len(), 2);
    }

    #[test]
    #[should_panic(expected = "Tensor type mismatch")]
    fn test_get_with_wrong_rank_has_clear_error() {
        let device = Device::default();
        let layer = layer(&device.clone().autodiff());
        let mut grads = GradientsParams::new();
        grads.register(layer.weight.id, Tensor::<2>::zeros([20, 20], &device));

        let _ = grads.get::<1>(layer.weight.id);
    }

    #[test]
    fn clip_norm_scales_all_gradients_by_their_combined_norm() {
        let device = Device::default();
        let layer = LinearConfig::new(2, 2).init(&device);
        let bias = layer.bias.as_ref().unwrap();
        let mut grads = GradientsParams::new();
        grads.register(
            layer.weight.id,
            Tensor::<2>::from_data([[3.0, 0.0], [0.0, 0.0]], &device),
        );
        grads.register(bias.id, Tensor::<1>::from_data([4.0, 0.0], &device));

        let norm = grads.clip_norm(&layer, 1.0).unwrap();

        // The combined norm is 5. Clipping each tensor alone would leave [1, 0] and [1, 0].
        norm.into_data()
            .assert_approx_eq::<f32>(&TensorData::from([5.0]), Tolerance::default());
        grads
            .get::<2>(layer.weight.id)
            .unwrap()
            .into_data()
            .assert_approx_eq::<f32>(
                &TensorData::from([[0.6, 0.0], [0.0, 0.0]]),
                Tolerance::default(),
            );
        grads
            .get::<1>(bias.id)
            .unwrap()
            .into_data()
            .assert_approx_eq::<f32>(&TensorData::from([0.8, 0.0]), Tolerance::default());
    }

    #[test]
    fn clip_norm_leaves_small_gradients_unchanged() {
        let device = Device::default();
        let layer = LinearConfig::new(2, 2).with_bias(false).init(&device);
        let mut grads = GradientsParams::new();
        grads.register(
            layer.weight.id,
            Tensor::<2>::from_data([[0.3, 0.4], [0.0, 0.0]], &device),
        );

        grads.clip_norm(&layer, 1.0).unwrap();

        grads
            .get::<2>(layer.weight.id)
            .unwrap()
            .into_data()
            .assert_approx_eq::<f32>(
                &TensorData::from([[0.3, 0.4], [0.0, 0.0]]),
                Tolerance::default(),
            );
        assert!(GradientsParams::new().clip_norm(&layer, 1.0).is_none());
    }

    #[test]
    fn clip_norm_reaches_lora_factors() {
        let device = Device::default().autodiff();
        let layer = layer(&device).apply_lora(burn::module::Lora::new(2, 4.0));
        let loss = layer.forward(random_tensor(&device)).sum();
        let mut grads = GradientsParams::from_grads(loss.backward(), &layer);
        let adapter = layer.weight.adapter().unwrap();
        let before = grads.get::<2>(adapter.b.id).unwrap();

        let norm = grads.clip_norm(&layer, 1e-3).unwrap().into_scalar::<f32>();

        assert!(norm > 1e-3, "the adapter has gradients to clip");
        let after = grads.get::<2>(adapter.b.id).unwrap();
        (before * (1e-3 / norm))
            .into_data()
            .assert_approx_eq::<f32>(&after.into_data(), Tolerance::default());
    }

    fn layer(device: &Device) -> Linear {
        LinearConfig::new(20, 20).init(device)
    }

    fn random_tensor(device: &Device) -> Tensor<2> {
        Tensor::<2>::random([2, 20], Distribution::Default, device)
    }
}
