//! Regression test for fusion broadcast bug with conv + manual BN.
//!
//! Bug: When two parallel branches each do conv(64→256) followed by manual
//! batch-norm (sub, div-by-sqrt, mul, add with [1,C,1,1] broadcast params),
//! the second branch produces wrong values under Fusion<f16>.
//!
//! The sqrt(v + eps) computation on the [1,256,1,1] parameter tensor gets
//! fused into the same kernel as the broadcast ops on the [1,256,16,16] conv output

use burn::backend::wgpu::{CubeBackend, WgpuRuntime};
use burn::nn::conv::Conv2dConfig;
use burn::prelude::*;
use burn_fusion::Fusion;

type Fused = Fusion<CubeBackend<WgpuRuntime, half::f16, i32, u8>>;
type Unfused = CubeBackend<WgpuRuntime, half::f16, i32, u8>;

const EPS: f64 = 1e-5;

fn pseudo_random(n: usize, seed: f32, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (i as f32 * seed + 0.7).sin() * scale)
        .collect()
}

fn make_conv<B: Backend>(
    in_ch: usize,
    out_ch: usize,
    seed: f32,
    dev: &B::Device,
) -> burn::nn::conv::Conv2d<B> {
    let conv = Conv2dConfig::new([in_ch, out_ch], [1, 1])
        .with_bias(false)
        .init::<B>(dev);
    let n = out_ch * in_ch;
    let scale = (2.0 / in_ch as f32).sqrt();
    let data = pseudo_random(n, seed, scale);
    let weight = Tensor::<B, 1>::from_floats(data.as_slice(), dev).reshape([out_ch, in_ch, 1, 1]);
    let mut record = conv.into_record();
    record.weight = burn::module::Param::initialized(burn::module::ParamId::new(), weight);
    Conv2dConfig::new([in_ch, out_ch], [1, 1])
        .with_bias(false)
        .init::<B>(dev)
        .load_record(record)
}

fn manual_bn<B: Backend>(
    x: Tensor<B, 4>,
    ch: usize,
    mean: &Tensor<B, 1>,
    var: &Tensor<B, 1>,
    gamma: &Tensor<B, 1>,
    beta: &Tensor<B, 1>,
) -> Tensor<B, 4> {
    let m = mean.clone().reshape([1, ch, 1, 1]);
    let v = var.clone().reshape([1, ch, 1, 1]);
    let g = gamma.clone().reshape([1, ch, 1, 1]);
    let b = beta.clone().reshape([1, ch, 1, 1]);
    let std = (v + EPS).sqrt();
    (x - m) / std * g + b
}

fn two_conv_bn_branches<B: Backend>() -> Vec<f32> {
    let dev = B::Device::default();
    let conv_a = make_conv::<B>(64, 256, 1.0, &dev);
    let conv_b = make_conv::<B>(64, 256, 2.0, &dev);

    let ch = 256;
    let make_params = |seed: f32| {
        let mean =
            Tensor::<B, 1>::from_floats(pseudo_random(ch, seed + 300.0, 0.5).as_slice(), &dev);
        let var_data: Vec<f32> = pseudo_random(ch, seed + 400.0, 0.3)
            .iter()
            .map(|v| 0.5 + v.abs())
            .collect();
        let var = Tensor::<B, 1>::from_floats(var_data.as_slice(), &dev);
        let gamma_data: Vec<f32> = pseudo_random(ch, seed + 100.0, 0.5)
            .iter()
            .map(|v| 1.0 + v * 0.1)
            .collect();
        let gamma = Tensor::<B, 1>::from_floats(gamma_data.as_slice(), &dev);
        let beta =
            Tensor::<B, 1>::from_floats(pseudo_random(ch, seed + 200.0, 0.3).as_slice(), &dev);
        (mean, var, gamma, beta)
    };
    let (mean_a, var_a, gamma_a, beta_a) = make_params(1.0);
    let (mean_b, var_b, gamma_b, beta_b) = make_params(2.0);
    B::sync(&dev).unwrap();

    let numel = 64 * 16 * 16;
    let data: Vec<f32> = (0..numel).map(|i| (i as f32 * 7.13).sin() * 0.5).collect();
    let input: Tensor<B, 4> =
        Tensor::<B, 1>::from_floats(data.as_slice(), &dev).reshape([1, 64, 16, 16]);

    let a = manual_bn(
        conv_a.forward(input.clone()),
        ch,
        &mean_a,
        &var_a,
        &gamma_a,
        &beta_a,
    );
    let b = manual_bn(
        conv_b.forward(input),
        ch,
        &mean_b,
        &var_b,
        &gamma_b,
        &beta_b,
    );
    let out = a + b;

    let data = out.into_data();
    let vals: &[half::f16] = data.as_slice().unwrap();
    vals.iter().map(|v| f32::from(*v)).collect()
}

/// This test was failing only on Vulkan+Fusion+f16
#[test]
fn fusion_f16_two_branch_conv_manual_bn() {
    let fused = two_conv_bn_branches::<Fused>();
    let reference = two_conv_bn_branches::<Unfused>();

    assert_eq!(fused.len(), reference.len(), "output length mismatch");

    let max_diff = fused
        .iter()
        .zip(reference.iter())
        .map(|(f, r)| (f - r).abs())
        .fold(0.0f32, f32::max);

    let first_bad = fused
        .iter()
        .zip(reference.iter())
        .enumerate()
        .find(|(_, (f, r))| (*f - *r).abs() > 0.01 || f.is_nan() || f.is_infinite());

    if let Some((idx, (f, r))) = first_bad {
        panic!(
            "fused vs unfused mismatch: max_diff={max_diff:.6}, first bad at idx={idx}/{}: fused={f} ref={r}",
            fused.len()
        );
    }
}
