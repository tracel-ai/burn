use burn_core::{self as burn, Tensor};

use burn_core::module::ParamId;
use burn_core::prelude::{Backend, DeviceOps};
use burn_core::tensor::Device;
use burn_core::tensor::backend::DeviceId;

use super::GradientsParams;
use crate::LearningRate;
use burn::module::AutodiffModule;
use burn::record::Record;
use burn::tensor::backend::AutodiffBackend;

#[derive(Default)]
/// Exposes multiple gradients for each parameter.
pub struct MultiGradientsParams {
    /// Each [GradientsParams] has its associated [DeviceId].
    pub grads: Vec<(GradientsParams, DeviceId)>,
}

impl MultiGradientsParams {
    /// Removes the gradients for the given [parameter id](ParamId).
    ///
    /// Potentially accumulates the gradients from multiple sources using a device associated with
    /// a parameter id. The same parameter will be accumulated using the same device during
    /// all training.
    pub fn remove<B: Backend, const D: usize>(
        &mut self,
        id: ParamId,
    ) -> Option<(Tensor<B, D>, Device<B>)> {
        let (mut tensor, device, index) = self.select(id)?;

        for (i, (grads, _)) in self.grads.iter_mut().enumerate() {
            if i == index {
                continue;
            }

            if let Some(grad) = grads.remove::<B, D>(id) {
                tensor = tensor + grad.to_device(&device);
            }
        }

        Some((tensor, device))
    }

    fn select<B: Backend, const D: usize>(
        &mut self,
        id: ParamId,
    ) -> Option<(Tensor<B, D>, Device<B>, usize)> {
        let id_val = id.val() as usize;
        for i in 0..self.grads.len() {
            let selected_device_index = (id_val + i) % self.grads.len();

            if let Some(acc) = self.grads[selected_device_index].0.remove::<B, D>(id) {
                let device_id = self.grads[selected_device_index].1;
                let device = <B::Device as DeviceOps>::from_id(device_id);
                return Some((acc.to_device(&device), device, selected_device_index));
            }
        }

        None
    }
}

/// General trait to optimize [module](AutodiffModule).
pub trait Optimizer<M, B>: Send + Clone
where
    M: AutodiffModule<B>,
    B: AutodiffBackend,
{
    /// Optimizer associative type to be used when saving and loading the state.
    type Record: Record<B>;

    /// Perform the optimizer step using the given learning rate and gradients.
    /// The updated module is returned.
    fn step(&mut self, lr: LearningRate, module: M, grads: GradientsParams) -> M;

    /// Perform the optimizer step using the given learning rate and gradients.
    /// The updated module is returned.
    fn step_multi(&mut self, lr: LearningRate, module: M, grads: MultiGradientsParams) -> M;

    /// Get the current state of the optimizer as a [record](Record).
    fn to_record(&self) -> Self::Record;

    /// Load the state of the optimizer as a [record](Record).
    fn load_record(self, record: Self::Record) -> Self;
}
