use super::scheduler::{tensor_from_vec, DiffusionScheduler};
use crate::tensor::{Distribution, Tensor, backend::Backend};

/// Configuration for the PingPong stochastic scheduler.
#[derive(Debug, Clone)]
pub struct FlowMatchPingPongConfig {
    /// Number of training timesteps used to scale `sigma <-> t`.
    pub num_train_timesteps: usize,
    /// Shift factor applied to the sigma schedule (default 1.0).
    pub shift: f32,
    /// Maximum sigma used to construct the schedule (default 1.0).
    pub sigma_max: f32,
}

impl Default for FlowMatchPingPongConfig {
    fn default() -> Self {
        Self { num_train_timesteps: 1000, shift: 1.0, sigma_max: 1.0 }
    }
}

/// PingPong flow-matching scheduler (stochastic variant).
pub struct FlowMatchPingPong<B: Backend, const D: usize> {
    cfg: FlowMatchPingPongConfig,
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
    step_index: usize,
    _phantom: core::marker::PhantomData<B>,
}

impl<B: Backend, const D: usize> FlowMatchPingPong<B, D> {
    pub fn new(cfg: FlowMatchPingPongConfig) -> Self {
        let mut s = Self { cfg, sigmas: Vec::new(), timesteps: Vec::new(), step_index: 0, _phantom: core::marker::PhantomData };
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
    }
}

impl<B: Backend, const D: usize> DiffusionScheduler<B, D> for FlowMatchPingPong<B, D> {
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
    }

    fn step_index(&self) -> usize { self.step_index }

    fn step(
        &mut self,
        model_output: Tensor<B, D>,
        _timestep: f32,
        sample: Tensor<B, D>,
        _omega: f32,
    ) -> Tensor<B, D> {
        let i = self.step_index;
        let sigma = self.sigmas[i];
        let sigma_next = self.sigmas[i + 1];

        // Stochastic ping-pong step: denoise then sample with new sigma.
        let denoised = sample.clone() - model_output.mul_scalar(sigma);
        let shape: [usize; D] = denoised.shape().dims();
        let noise = Tensor::<B, D>::random::<[usize; D]>(shape, Distribution::Default, &denoised.device());
        let prev_sample = denoised.mul_scalar(1.0 - sigma_next) + noise.mul_scalar(sigma_next);

        self.step_index += 1;
        prev_sample
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
