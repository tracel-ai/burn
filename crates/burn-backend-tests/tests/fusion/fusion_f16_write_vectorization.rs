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

use super::*;
use burn_tensor::{
    DType, Device, TensorCreationOptions, TensorData, Tolerance, module::conv2d, ops::ConvOptions,
};
use std::sync::{Arc, Mutex};

const CH: usize = 8;
const INIT: f64 = 0.02;
const EPS: f64 = 1e-5;

struct RunningState {
    value: Arc<Mutex<TestTensor<1>>>,
}

impl RunningState {
    fn new(value: TestTensor<1>) -> Self {
        Self {
            value: Arc::new(Mutex::new(value)),
        }
    }

    fn value(&self) -> TestTensor<1> {
        self.value.lock().unwrap().clone()
    }
}

/// Minimal clone of `BatchNorm` w/ `RunningState` for inference only.
struct BatchNorm {
    gamma: TestTensor<1>,
    beta: TestTensor<1>,
    running_mean: RunningState,
    running_var: RunningState,
}

impl BatchNorm {
    fn new(opts: TensorCreationOptions) -> Self {
        Self {
            gamma: TestTensor::<1>::ones([CH], opts.clone()),
            beta: TestTensor::<1>::zeros([CH], opts.clone()),
            running_mean: RunningState::new(TestTensor::<1>::zeros([CH], opts.clone())),
            running_var: RunningState::new(TestTensor::<1>::ones([CH], opts)),
        }
    }

    fn forward(&self, input: TestTensor<4>) -> TestTensor<4> {
        let device = input.device();
        let channels = input.dims()[1];
        let mean = self.running_mean.value().to_device(&device);
        let var = self.running_var.value().to_device(&device);

        let mean = mean.reshape([1, channels, 1, 1]);
        let var = var.reshape([1, channels, 1, 1]);
        let std = (var + EPS).sqrt();

        let x = (input - mean) / std;
        let x = x * self.gamma.clone().reshape([1, channels, 1, 1]);
        x + self.beta.clone().reshape([1, channels, 1, 1])
    }
}

fn make_conv_weight(opts: TensorCreationOptions) -> TestTensor<4> {
    TestTensor::full([CH, 3, 1, 1], INIT, opts)
}

fn two_branch_conv_bn_add(dev: Device, dtype: DType) -> TensorData {
    let opts: TensorCreationOptions = (&dev, dtype).into();
    let weight_a = make_conv_weight(opts.clone());
    let weight_b = make_conv_weight(opts.clone());
    let bn_a = BatchNorm::new(opts.clone());
    let bn_b = BatchNorm::new(opts.clone());

    dev.sync().unwrap();

    let input = TestTensor::<4>::ones([1, 3, 32, 32], opts) * 0.5;
    let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);
    let a = bn_a.forward(conv2d(input.clone(), weight_a, None, options.clone()));
    let b = bn_b.forward(conv2d(input, weight_b, None, options));
    (a + b).into_data()
}

/// Two parallel conv+BN branches added together — the Fusion<f16> result must
/// match reference.
///
/// This test was failing only on Vulkan+Fusion+f16
/// Reference: https://github.com/tracel-ai/burn/pull/4675
#[test]
fn fusion_f16_two_branch_conv_bn_add_matches_reference() {
    let fused_f16 = two_branch_conv_bn_add(Default::default(), DType::F16);
    let reference_f32 = two_branch_conv_bn_add(Default::default(), DType::F32);

    fused_f16
        .convert_dtype(DType::F32)
        .assert_approx_eq::<FloatElem>(&reference_f32, Tolerance::rel_abs(5e-3, 1e-2));
}
