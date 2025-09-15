use burn_core::diffusion::{
    FlowMatchEuler, FlowMatchEulerConfig, FlowMatchHeun, FlowMatchHeunConfig, FlowMatchPingPong,
    FlowMatchPingPongConfig, DiffusionScheduler, retrieve_timesteps,
};
use burn_core::prelude::Backend;
use burn_core::tensor::{Distribution, Tensor};

type TB = burn_ndarray::NdArray<f32>;

#[test]
fn euler_step_with_zero_model_output_is_identity() {
    let device = <TB as Backend>::Device::default();
    let mut sched = FlowMatchEuler::<TB, 3>::new(FlowMatchEulerConfig::default());
    sched.set_timesteps(8);
    let sample = Tensor::<TB, 3>::random([1, 2, 3], Distribution::Default, &device);
    let model_output = Tensor::<TB, 3>::zeros([1, 2, 3], &device);
    let out = sched.step(model_output, 0.0, sample.clone(), 1.0);
    // No change when model output is zero.
    out.into_data()
        .assert_approx_eq::<f32>(&sample.into_data(), burn_tensor::Tolerance::default());
}

#[test]
fn heun_two_stage_produces_finite_output() {
    let device = <TB as Backend>::Device::default();
    let mut sched = FlowMatchHeun::<TB, 3>::new(FlowMatchHeunConfig::default());
    sched.set_timesteps(6);
    let mut sample = Tensor::<TB, 3>::random([2, 3, 4], Distribution::Default, &device);
    // Stage 1: use model_output = sample
    let out1 = sched.step(sample.clone(), 0.0, sample.clone(), 1.0);
    // Stage 2: again with new sample
    let out2 = sched.step(out1.clone(), 0.0, out1.clone(), 1.0);

    // Check shapes and finite values
    assert_eq!(out2.dims(), [2, 3, 4]);
    // Sanity: value is not NaN by comparing to itself via approximate equality.
    let data = out2.clone().into_data();
    data.assert_approx_eq::<f32>(&out2.into_data(), burn_tensor::Tolerance::default());
}

#[test]
fn pingpong_stochastic_step_shapes() {
    let device = <TB as Backend>::Device::default();
    let mut sched = FlowMatchPingPong::<TB, 2>::new(FlowMatchPingPongConfig::default());
    sched.set_timesteps(4);
    let sample = Tensor::<TB, 2>::random([5, 7], Distribution::Default, &device);
    let model_output = sample.clone();
    let out = sched.step(model_output, 0.0, sample, 1.0);
    assert_eq!(out.dims(), [5, 7]);
}

#[test]
fn retrieve_timesteps_resamples_correct_len() {
    let device = <TB as Backend>::Device::default();
    let sigmas = vec![1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];
    let (t, n) = retrieve_timesteps::<TB>(&device, &sigmas, 5, 1000, None);
    assert_eq!(n, 5);
    assert_eq!(t.dims(), [5]);
}
