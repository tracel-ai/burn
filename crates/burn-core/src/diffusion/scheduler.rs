use crate::tensor::{Shape, Tensor, backend::Backend};

/// Trait implemented by diffusion schedulers used during inference.
///
/// The scheduler is responsible for producing a sequence of timesteps (or
/// corresponding sigmas) and for stepping the latent sample between them.
pub trait DiffusionScheduler<B: Backend, const D: usize> {
    /// Returns the sigmas used for the current inference schedule on device.
    fn sigmas(&self, device: &B::Device) -> Tensor<B, 1>;

    /// Returns the corresponding discrete timesteps `t = sigma * num_train_timesteps` on device.
    fn timesteps(&self, device: &B::Device) -> Tensor<B, 1>;

    /// Sets the number of inference steps and recomputes the internal schedule.
    fn set_timesteps(&mut self, num_inference_steps: usize);

    /// Number of training timesteps used to convert `sigma` to `t` and vice-versa.
    fn num_train_timesteps(&self) -> usize;

    /// Resets the internal step index to the beginning of the schedule.
    fn reset(&mut self);

    /// Current step index within the schedule.
    fn step_index(&self) -> usize;

    /// Advances the sample according to the scheduler dynamics.
    ///
    /// - `model_output`: model prediction at current sample and timestep
    /// - `timestep`: current discrete timestep value (one of `self.timesteps()`)
    /// - `sample`: current latent/sample to be advanced
    /// - `omega`: optional strength factor (used by some flow-matching variants); set to `1.0` if unsure.
    fn step(
        &mut self,
        model_output: Tensor<B, D>,
        timestep: f32,
        sample: Tensor<B, D>,
        omega: f32,
    ) -> Tensor<B, D>;

    /// Scales a clean sample by the forward process at the given timestep with additive noise.
    fn scale_noise(
        &mut self,
        sample: Tensor<B, D>,
        timestep: f32,
        noise: Tensor<B, D>,
    ) -> Tensor<B, D>;
}

/// Utility: allocate a 1D tensor on a backend from an iterator of f32 values.
pub(crate) fn tensor_from_vec<B: Backend>(device: &B::Device, values: &[f32]) -> Tensor<B, 1> {
    Tensor::from_floats(values, device)
}

