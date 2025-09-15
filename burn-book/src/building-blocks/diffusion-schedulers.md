# Diffusion Schedulers

This page introduces a small set of schedulers and utilities for diffusion model inference.
They provide a convenient, backend‑agnostic way to generate timesteps and to advance latent
samples using common ODE/SDE steppers.

## Why

- Flow‑matching and diffusion pipelines often share the same stepping logic (Euler / Heun,
  and simple stochastic variants). Providing a stable API in `burn-core` simplifies building
  pipelines across domains (audio, image, video) without duplicating boilerplate.

## API Overview

- `DiffusionScheduler<B, D>` trait
  - `set_timesteps(num_inference_steps)`: configure the schedule
  - `sigmas(&device) -> Tensor<_, 1>` and `timesteps(&device) -> Tensor<_, 1>`
  - `step(model_output, timestep, sample, omega) -> sample`: advance one step
  - `scale_noise(sample, timestep, noise) -> noisy_sample`: forward process helper

- Implementations
  - `FlowMatchEuler` — first‑order ODE step with optional mean‑shift scaling
  - `FlowMatchHeun` — second‑order (Heun) stepper for improved local accuracy
  - `FlowMatchPingPong` — simple stochastic step (denoise + resample noise)

- Utility
  - `retrieve_timesteps(device, sigmas, steps, num_train_timesteps, sigmas_override)`:
    resamples timesteps for an arbitrary number of inference steps, compatible with
    timesteps used by common diffusion libraries.

## Example

```rust
use burn::prelude::*;
use burn::diffusion::{FlowMatchEuler, FlowMatchEulerConfig, DiffusionScheduler};

let device = Default::default();
let mut sched = FlowMatchEuler::<burn_ndarray::NdArray<f32>, 4>::new(
    FlowMatchEulerConfig { num_train_timesteps: 1000, shift: 1.0, sigma_max: 1.0 }
);
sched.set_timesteps(50);

let sigmas = sched.sigmas(&device);
let t = sched.timesteps(&device);

// one step
let sample = Tensor::zeros([1, 8, 16, 16], &device);
let model_out = sample.clone();
let t0 = t.clone().into_data().convert::<f32>().value[0];
let next = sched.step(model_out, t0, sample, 1.0);
```

## References

- [Flow Matching for Generative Modeling (arXiv)](https://arxiv.org/abs/2210.02747)
- Diffusion scheduling patterns inspired by practical implementations in the community, e.g.
  [Hugging Face diffusers](https://github.com/huggingface/diffusers).

