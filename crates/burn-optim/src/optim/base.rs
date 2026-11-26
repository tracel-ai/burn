use burn_core::Tensor;

use burn_core::module::ParamId;
use burn_core::tensor::{DType, Device};

use super::GradientsParams;
use alloc::vec::Vec;

#[derive(Default)]
/// Exposes multiple gradients for each parameter.
pub struct MultiGradientsParams {
    /// Each [GradientsParams] has its associated [Device].
    pub grads: Vec<(GradientsParams, Device)>,
}

impl MultiGradientsParams {
    /// Removes the gradients for the given [parameter id](ParamId).
    ///
    /// Potentially accumulates the gradients from multiple sources using a device associated with
    /// a parameter id. The same parameter will be accumulated using the same device during
    /// all training.
    pub fn remove<const D: usize>(&mut self, id: ParamId) -> Option<(Tensor<D>, Device)> {
        self.remove_with(id, |lhs, rhs| lhs + rhs)
            .map(|(grad, device, _)| (grad, device))
    }

    /// Keep the first contribution's dtype so a widened sum can be rounded after clipping.
    pub(crate) fn remove_with<const D: usize>(
        &mut self,
        id: ParamId,
        add: impl Fn(Tensor<D>, Tensor<D>) -> Tensor<D>,
    ) -> Option<(Tensor<D>, Device, DType)> {
        let (mut tensor, device, index) = self.select(id)?;
        let dtype = tensor.dtype();

        for (i, (grads, _)) in self.grads.iter_mut().enumerate() {
            if i == index {
                continue;
            }

            if let Some(grad) = grads.remove::<D>(id) {
                tensor = add(tensor, grad.to_device(&device));
            }
        }

        Some((tensor, device, dtype))
    }

    fn select<const D: usize>(&mut self, id: ParamId) -> Option<(Tensor<D>, Device, usize)> {
        let id_val = id.val() as usize;
        for i in 0..self.grads.len() {
            let selected_device_index = (id_val + i) % self.grads.len();

            if let Some(acc) = self.grads[selected_device_index].0.remove::<D>(id) {
                let device = &self.grads[selected_device_index].1;
                return Some((acc.to_device(device), device.clone(), selected_device_index));
            }
        }

        None
    }
}
