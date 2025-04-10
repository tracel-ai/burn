use core::marker::PhantomData;

use crate::module::{AutodiffModule, ModuleVisitor, ParamId};

use burn_tensor::{Tensor, backend::AutodiffBackend, container::TensorContainerError};

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
    fn visit_float<const D: usize>(&mut self, id: ParamId, _tensor: &Tensor<B, D>) {
        fn fmt_message(
            id: ParamId,
            grad: &str,
            error: TensorContainerError,
        ) -> alloc::string::String {
            alloc::format!(
                "Failed to remove ID {} from {} due to an unexpected / unhandled error variant: {:?}",
                id,
                grad,
                error
            )
        }

        // Since we are explicitly removing the grad from the inner backend TensorContainerError::NotFound is the only error variant that should be encontered.
        let grad_updated = match self.grads_new.remove::<B::InnerBackend, D>(id) {
            Ok(new) => match self.grads.remove::<B::InnerBackend, D>(id) {
                Ok(grad) => grad.add(new),
                Err(error) => match error {
                    TensorContainerError::NotFound => new,
                    container_error => panic!("{}", fmt_message(id, "self.grads", container_error)),
                },
            },
            Err(new_grads_error) => match new_grads_error {
                TensorContainerError::NotFound => match self.grads.remove::<B::InnerBackend, D>(id)
                {
                    Ok(grad) => grad,
                    Err(error) => match error {
                        TensorContainerError::NotFound => return,
                        container_error => {
                            panic!("{}", fmt_message(id, "self.grads", container_error))
                        }
                    },
                },
                grads_new_error => panic!("{}", fmt_message(id, "self.grads_new", grads_new_error)),
            },
        };
        self.grads.register::<B::InnerBackend, D>(id, grad_updated);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        TestAutodiffBackend,
        nn::{Linear, LinearConfig},
    };
    use burn_tensor::{Distribution, backend::Backend};

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
