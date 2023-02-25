use crate::module::{Module, ModuleVisitor, ParamId};
use burn_tensor::{backend::ADBackend, Tensor};

use super::visitor::GradientsParams;

/// Accumulate gradients into a single [Gradients](ADBackend::Gradients) object.
pub struct GradientsAccumulator<B: ADBackend> {
    grads: GradientsParams<B>,
}

impl<B: ADBackend> Default for GradientsAccumulator<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: ADBackend> GradientsAccumulator<B> {
    /// Create a new gradients accumulator.
    pub fn new() -> Self {
        Self {
            grads: GradientsParams::<B>::new(),
        }
    }
}

impl<B: ADBackend> GradientsAccumulator<B> {
    /// Accumulate the given gradients for each parameter in the given module.
    pub fn accumulate<M>(&mut self, module: &M, grads: GradientsParams<B>)
    where
        M: Module<Backend = B>,
    {
        let mut visitor = ModuleGradsAccumulator::new(&mut self.grads, grads);
        module.visit(&mut visitor);
    }

    /// Return the accumulated gradients and reset the accumulator state.
    pub fn grads(&mut self) -> GradientsParams<B> {
        let mut grads = GradientsParams::<B>::new();
        core::mem::swap(&mut self.grads, &mut grads);

        grads
    }
}

#[derive(new)]
struct ModuleGradsAccumulator<'a, B: ADBackend> {
    grads: &'a mut GradientsParams<B>,
    grads_new: GradientsParams<B>,
}

impl<'a, B: ADBackend> ModuleVisitor<B> for ModuleGradsAccumulator<'a, B> {
    fn visit<const D: usize>(&mut self, id: &ParamId, _tensor: &Tensor<B, D>) {
        let grad_updated = match self.grads_new.get::<D>(id) {
            Some(new) => match self.grads.get::<D>(id) {
                Some(grad) => grad.add(new),
                None => new,
            },
            None => match self.grads.get::<D>(id) {
                Some(grad) => grad,
                None => return,
            },
        };

        self.grads.register(id.clone(), grad_updated);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        nn::{Linear, LinearConfig},
        optim::visitor::convert_grads,
        TestADBackend,
    };
    use burn_tensor::Distribution;

    #[test]
    fn test_accumulate_gradients_one_step() {
        let mut accumulator = GradientsAccumulator::<TestADBackend>::new();
        let layer = layer();
        let loss = layer.forward(random_tensor());
        let grads = convert_grads(loss.backward(), &layer);

        accumulator.accumulate(&layer, grads);

        let grads = accumulator.grads();
        assert!(!grads.is_empty())
    }

    #[test]
    fn test_accumulate_gradients_two_steps() {
        let mut accumulator = GradientsAccumulator::<TestADBackend>::new();
        let layer = layer();
        let loss_1 = layer.forward(random_tensor());
        let loss_2 = layer.forward(random_tensor());
        let grads_1 = convert_grads(loss_1.backward(), &layer);
        let grads_2 = convert_grads(loss_2.backward(), &layer);

        accumulator.accumulate(&layer, grads_1);
        accumulator.accumulate(&layer, grads_2);

        let grads = accumulator.grads();
        assert_eq!(grads.len(), 2)
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
