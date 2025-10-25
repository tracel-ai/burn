use burn_core as burn;

use core::marker::PhantomData;

use burn::module::{AutodiffModule, ModuleVisitor, Param};
use burn::tensor::{Tensor, backend::AutodiffBackend};

use super::GradientsParams;

/// Accumulate gradients into a single [Gradients](AutodiffBackend::Gradients) object.
pub struct GradientsAccumulator<M> {
    grads: GradientsParams,
    phantom: PhantomData<M>,
}

impl<M> Default for GradientsAccumulator<M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<M> GradientsAccumulator<M> {
    /// Create a new gradients accumulator.
    pub fn new() -> Self {
        Self {
            grads: GradientsParams::new(),
            phantom: PhantomData,
        }
    }
}

impl<M> GradientsAccumulator<M> {
    /// Accumulate the given gradients for each parameter in the given module.
    pub fn accumulate<B: AutodiffBackend>(&mut self, module: &M, grads: GradientsParams)
    where
        M: AutodiffModule<B>,
    {
        let mut visitor = ModuleGradsAccumulator::<M>::new(&mut self.grads, grads);
        module.visit(&mut visitor);
    }

    /// Return the accumulated gradients and reset the accumulator state.
    pub fn grads(&mut self) -> GradientsParams {
        let mut grads = GradientsParams::new();
        core::mem::swap(&mut self.grads, &mut grads);

        grads
    }
}

#[derive(new)]
struct ModuleGradsAccumulator<'a, M> {
    grads: &'a mut GradientsParams,
    grads_new: GradientsParams,
    phantom: PhantomData<M>,
}

impl<B: AutodiffBackend, M: AutodiffModule<B>> ModuleVisitor<B> for ModuleGradsAccumulator<'_, M> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let grad_updated = match self.grads_new.remove::<B::InnerBackend, D>(param.id) {
            Some(new) => match self.grads.remove::<B::InnerBackend, D>(param.id) {
                Some(grad) => grad.add(new),
                None => new,
            },
            None => match self.grads.remove::<B::InnerBackend, D>(param.id) {
                Some(grad) => grad,
                None => return,
            },
        };

        self.grads
            .register::<B::InnerBackend, D>(param.id, grad_updated);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestAutodiffBackend;
    use burn::tensor::{Distribution, backend::Backend};
    use burn_nn::{Linear, LinearConfig};

    #[test]
    fn test_accumulate_gradients_one_step() {
        let device = Default::default();
        let mut accumulator = GradientsAccumulator::new();
        let layer = layer::<TestAutodiffBackend>(&device);
        let loss = layer.forward(random_tensor::<TestAutodiffBackend>(&device));
        let grads = GradientsParams::from_grads(loss.backward(), &layer);

        accumulator.accumulate(&layer, grads);

        let grads = accumulator.grads();
        assert!(!grads.is_empty())
    }

    #[test]
    fn test_accumulate_gradients_two_steps() {
        let device = Default::default();
        let mut accumulator = GradientsAccumulator::new();
        let layer = layer::<TestAutodiffBackend>(&device);
        let loss_1 = layer.forward(random_tensor(&device));
        let loss_2 = layer.forward(random_tensor(&device));
        let grads_1 = GradientsParams::from_grads(loss_1.backward(), &layer);
        let grads_2 = GradientsParams::from_grads(loss_2.backward(), &layer);

        accumulator.accumulate(&layer, grads_1);
        accumulator.accumulate(&layer, grads_2);

        let grads = accumulator.grads();
        assert_eq!(grads.len(), 2)
    }

    fn layer<B: Backend>(device: &B::Device) -> Linear<B> {
        LinearConfig::new(20, 20).init(device)
    }

    fn random_tensor<B: Backend>(device: &B::Device) -> Tensor<B, 2> {
        Tensor::<B, 2>::random([2, 20], Distribution::Default, device)
    }
}
