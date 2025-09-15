use super::scheduler::{tensor_from_vec, DiffusionScheduler};
use super::utils::logistic_rescale;
use crate::tensor::{Tensor, backend::Backend};

/// Configuration for the Heun flow-matching scheduler.
#[derive(Debug, Clone)]
pub struct FlowMatchHeunConfig {
    /// Number of training timesteps used to scale `sigma <-> t`.
    pub num_train_timesteps: usize,
    /// Shift factor applied to the sigma schedule (default 1.0).
    pub shift: f32,
    /// Maximum sigma used to construct the schedule (default 1.0).
    pub sigma_max: f32,
}

impl Default for FlowMatchHeunConfig {
    fn default() -> Self {
        Self { num_train_timesteps: 1000, shift: 1.0, sigma_max: 1.0 }
    }
}

/// Heun (2nd-order) flow-matching scheduler.
pub struct FlowMatchHeun<B: Backend, const D: usize> {
    cfg: FlowMatchHeunConfig,
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
    step_index: usize,
    prev_derivative: Option<Tensor<B, D>>,
    dt: Option<f32>,
    sample_first: Option<Tensor<B, D>>,
}

impl<B: Backend, const D: usize> FlowMatchHeun<B, D> {
    pub fn new(cfg: FlowMatchHeunConfig) -> Self {
        let mut s = Self {
            cfg,
            sigmas: Vec::new(),
            timesteps: Vec::new(),
            step_index: 0,
            prev_derivative: None,
            dt: None,
            sample_first: None,
        };
        s.set_timesteps(50);
        s
    }

    fn build_schedule(&mut self, num_inference_steps: usize) {
        let n = self.cfg.num_train_timesteps as f32;
        let sigma_min = 1.0 / n;
        let sigma_max = self.cfg.sigma_max;

        let mut sigmas = Vec::with_capacity(num_inference_steps + 1);
        for i in 0..num_inference_steps {
            let a = i as f32 / (num_inference_steps.saturating_sub(1).max(1) as f32);
            let s = sigma_max * (1.0 - a) + sigma_min * a;
            let s = self.cfg.shift * s / (1.0 + (self.cfg.shift - 1.0) * s);
            sigmas.push(s);
        }
        sigmas.push(0.0);

        let timesteps = sigmas
            .iter()
            .map(|s| s * self.cfg.num_train_timesteps as f32)
            .collect::<Vec<_>>();

        self.sigmas = sigmas;
        self.timesteps = timesteps;
        self.step_index = 0;
        self.prev_derivative = None;
        self.dt = None;
        self.sample_first = None;
    }
}

impl<B: Backend, const D: usize> DiffusionScheduler<B, D> for FlowMatchHeun<B, D> {
    fn sigmas(&self, device: &B::Device) -> Tensor<B, 1> {
        tensor_from_vec(device, &self.sigmas)
    }

    fn timesteps(&self, device: &B::Device) -> Tensor<B, 1> {
        tensor_from_vec(device, &self.timesteps)
    }

    fn set_timesteps(&mut self, num_inference_steps: usize) {
        self.build_schedule(num_inference_steps);
    }

    fn num_train_timesteps(&self) -> usize {
        self.cfg.num_train_timesteps
    }

    fn reset(&mut self) {
        self.step_index = 0;
        self.prev_derivative = None;
        self.dt = None;
        self.sample_first = None;
    }

    fn step_index(&self) -> usize {
        self.step_index
    }

    fn step(
        &mut self,
        model_output: Tensor<B, D>,
        _timestep: f32,
        sample: Tensor<B, D>,
        omega: f32,
    ) -> Tensor<B, D> {
        let i = self.step_index;
        let (sigma, sigma_next) = if self.prev_derivative.is_none() {
            (self.sigmas[i], self.sigmas[i + 1])
        } else {
            // 2nd order path reuses sigma_next for averaging.
            (self.sigmas[i - 1], self.sigmas[i])
        };

        let omega = logistic_rescale(omega, 0.9, 1.1, 0.0, 0.1);

        let prev = if self.prev_derivative.is_none() {
            // First (Euler) stage.
            // Derivative estimate in ODE form.
            let denoised = sample.clone() - model_output.clone().mul_scalar(sigma);
            let derivative = (sample.clone() - denoised).div_scalar(sigma.max(1e-8));
            let dt = sigma_next - sigma;
            self.prev_derivative = Some(derivative.clone());
            self.dt = Some(dt);
            self.sample_first = Some(sample.clone());

            let dx = derivative.mul_scalar(dt * omega);
            sample + dx
        } else {
            // Second (Heun) stage.
            let sample_first = self.sample_first.take().unwrap();
            let dt = self.dt.take().unwrap();
            let prev_derivative = self.prev_derivative.take().unwrap();

            let denoised = sample.clone() - model_output.clone().mul_scalar(sigma_next);
            let derivative_next = (sample - denoised).div_scalar(sigma_next.max(1e-8));
            let derivative = prev_derivative.mul_scalar(0.5) + derivative_next.mul_scalar(0.5);

            let dx = derivative.mul_scalar(dt * omega);
            sample_first + dx
        };

        self.step_index += 1;
        prev
    }

    fn scale_noise(
        &mut self,
        sample: Tensor<B, D>,
        timestep: f32,
        noise: Tensor<B, D>,
    ) -> Tensor<B, D> {
        let idx = self
            .timesteps
            .iter()
            .position(|t| (*t - timestep).abs() < 1e-3)
            .unwrap_or(self.step_index);
        let sigma = self.sigmas[idx];
        noise.mul_scalar(sigma).add(sample.mul_scalar(1.0 - sigma))
    }
}
