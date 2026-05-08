//! Regression test for fusion broadcast bug with conv + manual BN.
//!
//! Bug: When two parallel branches each do conv(64→256) followed by manual
//! batch-norm (sub, div-by-sqrt, mul, add with [1,C,1,1] broadcast params),
//! the second branch produces wrong values under Fusion<f16>.
//!
//! The sqrt(v + eps) computation on the [1,256,1,1] parameter tensor gets
//! fused into the same kernel as the broadcast ops on the [1,256,16,16] conv output

use super::*;
use burn_tensor::{
    DType, Device, TensorCreationOptions, TensorData, Tolerance, module::conv2d, ops::ConvOptions,
};

const EPS: f64 = 1e-5;

fn pseudo_random(n: usize, seed: f32, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (i as f32 * seed + 0.7).sin() * scale)
        .collect()
}

fn make_conv_weight(
    in_ch: usize,
    out_ch: usize,
    seed: f32,
    opts: TensorCreationOptions,
) -> TestTensor<4> {
    // kernel_size=[1, 1]; stride=[1, 1]; dilation=[1, 1]; groups=1;
    let n = out_ch * in_ch;
    let scale = (2.0 / in_ch as f32).sqrt();
    let data = TensorData::new(pseudo_random(n, seed, scale), [out_ch, in_ch, 1, 1]);
    TestTensor::from_data(data, opts)
}

fn manual_bn(
    x: TestTensor<4>,
    ch: usize,
    mean: &TestTensor<1>,
    var: &TestTensor<1>,
    gamma: &TestTensor<1>,
    beta: &TestTensor<1>,
) -> TestTensor<4> {
    let m = mean.clone().reshape([1, ch, 1, 1]);
    let v = var.clone().reshape([1, ch, 1, 1]);
    let g = gamma.clone().reshape([1, ch, 1, 1]);
    let b = beta.clone().reshape([1, ch, 1, 1]);
    let std = (v + EPS).sqrt();
    (x - m) / std * g + b
}

fn two_conv_bn_branches(dev: Device, dtype: DType) -> TensorData {
    let opts: TensorCreationOptions = (&dev, dtype).into();
    let weight_a = make_conv_weight(64, 256, 1.0, opts.clone());
    let weight_b = make_conv_weight(64, 256, 2.0, opts.clone());

    let ch = 256;
    let make_params = |seed: f32| {
        let mean = TestTensor::<1>::from_data(
            TensorData::new(pseudo_random(ch, seed + 300.0, 0.5), [ch]),
            opts.clone(),
        );
        let var_data: Vec<f32> = pseudo_random(ch, seed + 400.0, 0.3)
            .iter()
            .map(|v| 0.5 + v.abs())
            .collect();
        let var = TestTensor::<1>::from_data(TensorData::new(var_data, [ch]), opts.clone());
        let gamma_data: Vec<f32> = pseudo_random(ch, seed + 100.0, 0.5)
            .iter()
            .map(|v| 1.0 + v * 0.1)
            .collect();
        let gamma = TestTensor::<1>::from_data(TensorData::new(gamma_data, [ch]), opts.clone());
        let beta = TestTensor::<1>::from_data(
            TensorData::new(pseudo_random(ch, seed + 200.0, 0.3), [ch]),
            opts.clone(),
        );
        (mean, var, gamma, beta)
    };
    let (mean_a, var_a, gamma_a, beta_a) = make_params(1.0);
    let (mean_b, var_b, gamma_b, beta_b) = make_params(2.0);
    dev.sync().unwrap();

    let numel = 64 * 16 * 16;
    let data: Vec<f32> = (0..numel).map(|i| (i as f32 * 7.13).sin() * 0.5).collect();
    let input = TestTensor::<4>::from_data(TensorData::new(data, [1, 64, 16, 16]), opts);

    let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);
    let conv_a_out = conv2d(input.clone(), weight_a, None, options.clone());
    let conv_b_out = conv2d(input, weight_b, None, options);

    let a = manual_bn(conv_a_out, ch, &mean_a, &var_a, &gamma_a, &beta_a);
    let b = manual_bn(conv_b_out, ch, &mean_b, &var_b, &gamma_b, &beta_b);
    let out = a + b;

    out.into_data()
}

/// This test was failing only on Vulkan+Fusion+f16
/// Reference: https://github.com/tracel-ai/burn/pull/4675
#[test]
fn fusion_f16_two_branch_conv_manual_bn() {
    let fused_f16 = two_conv_bn_branches(Default::default(), DType::F16);
    let reference_f32 = two_conv_bn_branches(Default::default(), DType::F32);

    fused_f16
        .convert_dtype(DType::F32)
        .assert_approx_eq::<FloatElem>(&reference_f32, Tolerance::rel_abs(5e-3, 1e-2));
}
