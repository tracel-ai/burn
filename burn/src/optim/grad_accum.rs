use crate::module::{Module, ModuleVisitor, ParamId};
use burn_tensor::{
    backend::{ADBackend, Gradients},
    Tensor,
};

/// Accumulate gradients into a single [Gradients](ADBackend::Gradients) object.
pub struct GradientsAccumulator<B: ADBackend> {
    grads: B::Gradients,
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
            grads: B::Gradients::empty(),
        }
    }
}

impl<B: ADBackend> GradientsAccumulator<B> {
    /// Accumulate the given gradients for each parameter in the given module.
    pub fn accumulate<M>(&mut self, module: &M, grads: &B::Gradients)
    where
        M: Module<Backend = B>,
    {
        let mut visitor = ModuleGradsAccumulator::new(&mut self.grads, grads);
        module.visit(&mut visitor);
    }

    /// Return the accumulated gradients and reset the accumulator state.
    pub fn grads(&mut self) -> Option<B::Gradients> {
        let mut grads = B::Gradients::empty();
        std::mem::swap(&mut self.grads, &mut grads);

        Some(grads)
    }
}

#[derive(new)]
struct ModuleGradsAccumulator<'a, B: ADBackend> {
    grads: &'a mut B::Gradients,
    grads_new: &'a B::Gradients,
}

impl<'a, B: ADBackend> ModuleVisitor<B> for ModuleGradsAccumulator<'a, B> {
    fn visit<const D: usize>(&mut self, _id: &ParamId, tensor: &Tensor<B, D>) {
        let grad_updated = match tensor.grad(self.grads_new) {
            Some(new) => match tensor.grad(&self.grads) {
                Some(grad) => grad.add(&new),
                None => new.clone(),
            },
            None => match tensor.grad(&self.grads) {
                Some(grad) => grad.clone(),
                None => return,
            },
        };

        self.grads.register(tensor.node_id(), grad_updated);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        nn::{Linear, LinearConfig},
        TestADBackend,
    };
    use burn_tensor::Distribution;

    #[test]
    fn test_accumulate_gradients() {
        let mut accumulator = GradientsAccumulator::<TestADBackend>::new();
        let layer = layer();
        let loss = layer.forward(random_tensor());
        let grads = loss.backward();

        accumulator.accumulate(&layer, &grads);

        let grads = accumulator.grads().unwrap();
        assert!(!Gradients::<TestADBackend>::is_empty(&grads))
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
