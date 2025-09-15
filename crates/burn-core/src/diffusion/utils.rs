use crate::tensor::{Tensor, backend::Backend};

/// Returns `(timesteps, num_inference_steps)` as a 1D tensor on the given device.
///
/// If `sigmas_override` is provided, it is used to build `timesteps = sigmas * num_train_timesteps`.
pub fn retrieve_timesteps<B: Backend>(
    device: &B::Device,
    sigmas: &[f32],
    num_inference_steps: usize,
    num_train_timesteps: usize,
    sigmas_override: Option<&[f32]>,
) -> (Tensor<B, 1>, usize) {
    let used_sigmas: Vec<f32> = if let Some(s) = sigmas_override {
        s.to_vec()
    } else {
        // Resample evenly across the provided sigma schedule to match `num_inference_steps`.
        if num_inference_steps >= sigmas.len() {
            sigmas.to_vec()
        } else {
            // Linearly interpolate indices across the schedule.
            let last = sigmas.len() - 1;
            (0..num_inference_steps)
                .map(|i| {
                    let idx = (i as f32) * (last as f32) / ((num_inference_steps - 1).max(1) as f32);
                    let low = idx.floor() as usize;
                    let high = idx.ceil() as usize;
                    if low == high {
                        sigmas[low]
                    } else {
                        let w = idx - (low as f32);
                        sigmas[low] * (1.0 - w) + sigmas[high] * w
                    }
                })
                .collect()
        }
    };

    let timesteps_vals: Vec<f32> = used_sigmas
        .iter()
        .map(|s| s * num_train_timesteps as f32)
        .collect();
    (Tensor::from_floats(timesteps_vals.as_slice(), device), used_sigmas.len())
}

/// Logistic rescale helper used by some flow-matching variants.
pub(crate) fn logistic_rescale(x: f32, l: f32, u: f32, x0: f32, k: f32) -> f32 {
    l + (u - l) / (1.0 + (-k * (x - x0)).exp())
}
