//! Regression test for fusion kernel vector_size mismatch on output writes.
//!
//! Bug: When a fused kernel's computation width (config.width) differs from an
//! output tensor's registered vector_size, the SPIR-V store uses the wrong type
//! (e.g. vec4<f16> into a scalar f16 slot), corrupting adjacent memory.
//!
//! This manifests as Inf/NaN values in downstream tensors.  Only triggers with
//! f16 because f32 tensors tend to all receive the same vectorization on WGPU.
//!
//! The minimal trigger is two parallel conv+BatchNorm branches whose results
//! are added: the BatchNorm inplace running-stats outputs are alias tensors
//! registered with vector_size=1, while the fused block runs at width=4.

use burn::backend::wgpu::{CubeBackend, WgpuRuntime};
use burn::module::Initializer;
use burn::nn::BatchNormConfig;
use burn::nn::conv::Conv2dConfig;
use burn::prelude::*;
use burn_fusion::Fusion;

type Fused = Fusion<CubeBackend<WgpuRuntime, half::f16, i32, u8>>;
type Unfused = CubeBackend<WgpuRuntime, half::f16, i32, u8>;

const CH: usize = 8;
const INIT: Initializer = Initializer::Constant { value: 0.02 };
const EPS: f64 = 1e-5;

/// Runs conv+BN on two parallel branches and adds the results.
/// Generic over backend so we can compare Fusion vs non-Fusion output.
fn two_branch_conv_bn_add<B: Backend>(dev: &B::Device) -> Vec<f32> {
    let conv_a = Conv2dConfig::new([3, CH], [1, 1])
        .with_bias(false)
        .with_initializer(INIT)
        .init::<B>(dev);
    let bn_a = BatchNormConfig::new(CH).with_epsilon(EPS).init(dev);

    let conv_b = Conv2dConfig::new([3, CH], [1, 1])
        .with_bias(false)
        .with_initializer(INIT)
        .init::<B>(dev);
    let bn_b: burn::nn::BatchNorm<B> = BatchNormConfig::new(CH).with_epsilon(EPS).init(dev);

    B::sync(dev).unwrap();

    let input: Tensor<B, 4> = Tensor::ones([1, 3, 32, 32], dev) * 0.5;
    let a = bn_a.forward(conv_a.forward(input.clone()));
    let b = bn_b.forward(conv_b.forward(input));
    let out = a + b;

    let data = out.into_data();
    let vals: &[half::f16] = data.as_slice().unwrap();
    vals.iter().map(|v| f32::from(*v)).collect()
}

/// Two parallel conv+BN branches added together — the Fusion<f16> result must
/// match the non-fused reference.
///
/// This test was failing only on Vulkan+Fusion+f16
#[test]
fn fusion_f16_two_branch_conv_bn_add_matches_reference() {
    let fused = two_branch_conv_bn_add::<Fused>(&Default::default());
    let reference = two_branch_conv_bn_add::<Unfused>(&Default::default());

    assert_eq!(fused.len(), reference.len(), "output length mismatch");

    for (i, (f, r)) in fused.iter().zip(reference.iter()).enumerate() {
        assert!(
            (f - r).abs() < 1e-3,
            "mismatch at element {i}: fused={f} reference={r}"
        );
        assert!(
            f.is_finite(),
            "fused output contains non-finite value at element {i}: {f}"
        );
    }
}
