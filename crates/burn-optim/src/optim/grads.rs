use burn_core as burn;

use burn::{
    Tensor,
    tensor::{
        backend::{AutodiffBackend, Backend},
        container::{TensorContainer, TensorContainerError},
    },
};
#[cfg(feature = "collective")]
use burn_collective::{CollectiveError, PeerId, ReduceOperation, all_reduce};

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
    pub fn from_grads<B: AutodiffBackend, M: AutodiffModule<B>>(
        grads: B::Gradients,
        module: &M,
    ) -> Self {
        let mut grads = grads;
        Self::from_module(&mut grads, module)
    }

    /// Extract each tensor gradients for the given [module](AutodiffModule).
    pub fn from_module<B: AutodiffBackend, M: AutodiffModule<B>>(
        grads: &mut B::Gradients,
        module: &M,
    ) -> Self {
        let mut grads_params = GradientsParams::new();
        let mut visitor = GradientsParamsConverter::<M, B>::new(grads, &mut grads_params, None);
        module.visit(&mut visitor);
        grads_params
    }

    /// Extract tensor gradients for the given [module](AutodiffModule) and given parameters.
    pub fn from_params<B: AutodiffBackend, M: AutodiffModule<B>>(
        grads: &mut B::Gradients,
        module: &M,
        params: &[ParamId],
    ) -> Self {
        let mut grads_params = GradientsParams::new();
        let mut visitor =
            GradientsParamsConverter::<M, B>::new(grads, &mut grads_params, Some(params.to_vec()));
        module.visit(&mut visitor);
        grads_params
    }

    /// Get the gradients for the given [parameter id](ParamId).
    ///
    /// Returns `None` if no gradient is registered for the given id. Panics with a
    /// descriptive message if a gradient is registered for a different backend than `B`
    /// (most commonly: passing `B: AutodiffBackend` instead of `B::InnerBackend`).
    ///
    /// # Notes
    ///
    /// You should use [remove](GradientsParams::remove) if you want to get the gradients
    /// only one time.
    pub fn get<B, const D: usize>(&self, id: ParamId) -> Option<Tensor<B, D>>
    where
        B: Backend,
    {
        match self.container.get(&id) {
            Ok(primitive) => Some(Tensor::from_primitive(primitive)),
            Err(TensorContainerError::NotFound { .. }) => None,
            Err(e @ TensorContainerError::TypeMismatch { .. }) => panic!("{e}"),
        }
    }

    /// Remove the gradients for the given [parameter id](ParamId).
    ///
    /// Returns `None` if no gradient is registered for the given id. Panics with a
    /// descriptive message if a gradient is registered for a different backend than `B`.
    pub fn remove<B, const D: usize>(&mut self, id: ParamId) -> Option<Tensor<B, D>>
    where
        B: Backend,
    {
        match self.container.remove(&id) {
            Ok(primitive) => Some(Tensor::from_primitive(primitive)),
            Err(TensorContainerError::NotFound { .. }) => None,
            Err(e @ TensorContainerError::TypeMismatch { .. }) => panic!("{e}"),
        }
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
    pub fn to_device<B: AutodiffBackend, M: AutodiffModule<B>>(
        mut self,
        device: &B::Device,
        module: &M,
    ) -> Self {
        let mut visitor = GradientsParamsChangeDevice::<M, B>::new(device, &mut self);
        module.visit(&mut visitor);
        self
    }

    /// Syncs the gradient params with the other peers in the collective.
    #[cfg(feature = "collective")]
    pub fn all_reduce<B: Backend>(
        mut self,
        peer_id: PeerId,
        op: ReduceOperation,
    ) -> Result<Self, CollectiveError> {
        let mut ids = self
            .container
            .ids()
            .into_iter()
            .copied()
            .collect::<Vec<ParamId>>();
        // This is crucial, since the all-reduce operations need to happen in the same order for the same parameters on all nodes!
        ids.sort();

        for id in ids {
            let grad = match self.container.remove::<B>(&id) {
                Ok(grad) => grad,
                Err(TensorContainerError::NotFound { .. }) => {
                    // Was just observed in the freshly-collected `ids` snapshot above.
                    unreachable!("id present in ids() but missing from container");
                }
                Err(e @ TensorContainerError::TypeMismatch { .. }) => panic!("{e}"),
            };

            let grad = match grad {
                burn::tensor::TensorPrimitive::Float(grad) => {
                    let grad = all_reduce::<B>(peer_id, grad, op)?;
                    burn::tensor::TensorPrimitive::Float(grad)
                }
                burn::tensor::TensorPrimitive::QFloat(_grad) => {
                    unimplemented!("quantized all-reduce unimplemented")
                }
            };

            self.container.register::<B>(id, grad);
        }

        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TestAutodiffBackend, TestBackend};
    use burn::module::{Module, list_param_ids};
    use burn::tensor::{Distribution, backend::Backend};
    use burn_nn::{Linear, LinearConfig};

    #[test]
    fn test_convert_grads() {
        let device = Default::default();
        let layer_1 = layer::<TestAutodiffBackend>(&device);
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

    /// Regression test for #2924 / #3969 — the happy path.
    /// Gradients keyed under the inner backend can be retrieved by passing the inner backend.
    #[test]
    fn get_with_inner_backend_returns_gradient() {
        let device = Default::default();
        let layer = layer::<TestAutodiffBackend>(&device);
        let loss = layer.forward(random_tensor(&device));
        let grads = GradientsParams::from_grads(loss.backward(), &layer);

        let weight_id = layer.weight.id;
        let weight_grad = grads.get::<TestBackend, 2>(weight_id);
        assert!(weight_grad.is_some());
        assert_eq!(weight_grad.unwrap().dims(), [20, 20]);
    }

    /// Regression test for #2924 / #3969 — the bug case.
    /// Passing the autodiff backend instead of its inner backend used to panic with the
    /// generic `Option::unwrap()` message; now panics with a descriptive message naming
    /// the type mismatch and pointing at the inner-backend convention.
    #[test]
    #[should_panic(expected = "type mismatch")]
    fn get_with_autodiff_backend_panics_descriptively() {
        let device = Default::default();
        let layer = layer::<TestAutodiffBackend>(&device);
        let loss = layer.forward(random_tensor(&device));
        let grads = GradientsParams::from_grads(loss.backward(), &layer);

        // The bug-trigger: pass the autodiff backend instead of the inner backend.
        let _ = grads.get::<TestAutodiffBackend, 2>(layer.weight.id);
    }

    /// Regression test for #2924 / #3969 — same as `get_with_autodiff_backend_panics_descriptively`
    /// but on the `remove` path. Both code paths used to share the same `Option::unwrap()` bug.
    #[test]
    #[should_panic(expected = "type mismatch")]
    fn remove_with_autodiff_backend_panics_descriptively() {
        let device = Default::default();
        let layer = layer::<TestAutodiffBackend>(&device);
        let loss = layer.forward(random_tensor(&device));
        let mut grads = GradientsParams::from_grads(loss.backward(), &layer);

        let _ = grads.remove::<TestAutodiffBackend, 2>(layer.weight.id);
    }

    /// `get` for a parameter id that has no registered gradient still returns `None`,
    /// distinguishing the missing-entry case from the type-mismatch case.
    #[test]
    fn get_missing_id_returns_none() {
        let device = Default::default();
        let layer_1 = layer::<TestAutodiffBackend>(&device);
        let layer_2 = layer::<TestAutodiffBackend>(&device); // different param ids
        let loss = layer_1.forward(random_tensor(&device));
        let grads = GradientsParams::from_grads(loss.backward(), &layer_1);

        // layer_2's weight id is unknown to grads (only layer_1's grads are registered).
        let result = grads.get::<TestBackend, 2>(layer_2.weight.id);
        assert!(result.is_none());
    }

    /// Adversarial: a wrong-backend `TensorContainer::remove` MUST leave the entry intact
    /// rather than silently dropping it. The pre-fix code used `HashMap::remove` followed by
    /// `Box::downcast().unwrap()` — on a type mismatch the entry was already gone from the
    /// HashMap when the unwrap panicked, leaking the tensor into the failure path. The fix
    /// peeks the type via `Any::is::<T>()` first and only commits to removal once the
    /// downcast is guaranteed to succeed.
    #[test]
    fn container_remove_type_mismatch_leaves_entry_in_place() {
        use burn::tensor::backend::AutodiffBackend;
        use burn::tensor::container::{TensorContainer, TensorContainerError};
        type Inner = <TestAutodiffBackend as AutodiffBackend>::InnerBackend;

        let device = Default::default();
        let mut container = TensorContainer::<u64>::new();
        let tensor = Tensor::<Inner, 2>::random([3, 3], Distribution::Default, &device);
        container.register::<Inner>(42, tensor.into_primitive());
        assert_eq!(container.len(), 1);

        // Wrong backend → Err(TypeMismatch), entry MUST remain in the container.
        let result = container.remove::<TestAutodiffBackend>(&42);
        assert!(
            matches!(result, Err(TensorContainerError::TypeMismatch { .. })),
            "wrong-backend remove should return TypeMismatch, got {result:?}"
        );
        assert_eq!(
            container.len(),
            1,
            "entry must NOT be removed on type mismatch — original code leaked here"
        );

        // Correct-backend remove on the same id still works.
        let result = container.remove::<Inner>(&42);
        assert!(result.is_ok());
        assert_eq!(container.len(), 0);
    }

    /// Adversarial: repeated remove of the same id returns `Err(NotFound)` the second time,
    /// not a panic and not a stale tensor.
    #[test]
    fn container_remove_twice_returns_not_found() {
        use burn::tensor::backend::AutodiffBackend;
        use burn::tensor::container::{TensorContainer, TensorContainerError};
        type Inner = <TestAutodiffBackend as AutodiffBackend>::InnerBackend;

        let device = Default::default();
        let mut container = TensorContainer::<u64>::new();
        let tensor = Tensor::<Inner, 2>::random([2, 2], Distribution::Default, &device);
        container.register::<Inner>(7, tensor.into_primitive());

        assert!(container.remove::<Inner>(&7).is_ok());
        let result = container.remove::<Inner>(&7);
        assert!(matches!(result, Err(TensorContainerError::NotFound { .. })));
    }

    /// Adversarial: `get` (not `remove`) on a wrong backend returns `Err(TypeMismatch)`
    /// and does not consume the entry — multiple wrong-backend reads in a row stay
    /// consistent.
    #[test]
    fn container_get_type_mismatch_is_idempotent() {
        use burn::tensor::backend::AutodiffBackend;
        use burn::tensor::container::{TensorContainer, TensorContainerError};
        type Inner = <TestAutodiffBackend as AutodiffBackend>::InnerBackend;

        let device = Default::default();
        let mut container = TensorContainer::<u64>::new();
        let tensor = Tensor::<Inner, 2>::random([2, 2], Distribution::Default, &device);
        container.register::<Inner>(99, tensor.into_primitive());

        for _ in 0..3 {
            let result = container.get::<TestAutodiffBackend>(&99);
            assert!(matches!(result, Err(TensorContainerError::TypeMismatch { .. })));
        }
        assert_eq!(container.len(), 1);
        assert!(container.get::<Inner>(&99).is_ok());
    }

    fn layer<B: Backend>(device: &B::Device) -> Linear<B> {
        LinearConfig::new(20, 20).init(device)
    }

    fn random_tensor<B: Backend>(device: &B::Device) -> Tensor<B, 2> {
        Tensor::<B, 2>::random([2, 20], Distribution::Default, device)
    }
}
