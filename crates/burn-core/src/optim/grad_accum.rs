use core::marker::PhantomData;

use crate::module::{AutodiffModule, ModuleVisitor, ParamId};

use burn_tensor::{backend::AutodiffBackend, Tensor};

use super::GradientsParams;

/// Accumulate gradients into a single [Gradients](AutodiffBackend::Gradients) object.
pub struct GradientsAccumulator<M, P> {
    grads: GradientsParams<P>,
    phantom: PhantomData<M>,
}

impl<M, P> Default for GradientsAccumulator<M, P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<M, P> GradientsAccumulator<M, P> {
    /// Create a new gradients accumulator.
    pub fn new() -> Self {
        Self {
            grads: GradientsParams::new(),
            phantom: PhantomData,
        }
    }

    /// Accumulate the given gradients for each parameter in the given module.
    pub fn accumulate<B: AutodiffBackend<DynTensorPrimitive = P>>(
        &mut self,
        module: &M,
        grads: GradientsParams<P>,
    ) where
        M: AutodiffModule<B>,
    {
        let mut visitor = ModuleGradsAccumulator::<M, P>::new(&mut self.grads, grads);
        module.visit(&mut visitor);
    }

    /// Return the accumulated gradients and reset the accumulator state.
    pub fn grads(&mut self) -> GradientsParams<P> {
        let mut grads = GradientsParams::new();
        core::mem::swap(&mut self.grads, &mut grads);

        grads
    }
}

#[derive(new)]
struct ModuleGradsAccumulator<'a, M, P> {
    grads: &'a mut GradientsParams<P>,
    grads_new: GradientsParams<P>,
    phantom: PhantomData<M>,
}

impl<'a, B: AutodiffBackend, M: AutodiffModule<B>> ModuleVisitor<B>
    for ModuleGradsAccumulator<'a, M, B::DynTensorPrimitive>
{
    fn visit_float<const D: usize>(&mut self, id: &ParamId, _tensor: &Tensor<B, D>) {
        if let Some(new_grad) = self.grads_new.get::<B, D>(id) {
            self.grads.add(id.clone(), new_grad);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        nn::{Linear, LinearConfig},
        TestAutodiffBackend,
    };
    use burn_tensor::{backend::Backend, Distribution};

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
        LinearConfig::new(20, 20).with_bias(true).init(device)
    }

    fn random_tensor<B: Backend>(device: &B::Device) -> Tensor<B, 2> {
        Tensor::<B, 2>::random([2, 20], Distribution::Default, device)
    }
}
