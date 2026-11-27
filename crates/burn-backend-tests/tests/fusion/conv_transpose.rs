use super::*;
use burn_fusion::inspect::FusionInspector;
use burn_ir::{
    ModuleOperationIr::{
        ConvTranspose1dBiasBackward, ConvTranspose2dBiasBackward, ConvTranspose3dBiasBackward,
    },
    OperationIr::{self, Module},
};
use burn_tensor::{
    Device, TensorData,
    module::{conv_transpose1d, conv_transpose2d, conv_transpose3d},
    ops::ConvTransposeOptions,
};

#[test]
fn conv_transpose1d_bias_backward_forwards_to_backend() {
    assert_bias_forwarded(conv_transpose1d, |op| {
        matches!(op, Module(ConvTranspose1dBiasBackward(_)))
    });
}

#[test]
fn conv_transpose2d_bias_backward_forwards_to_backend() {
    assert_bias_forwarded(conv_transpose2d, |op| {
        matches!(op, Module(ConvTranspose2dBiasBackward(_)))
    });
}

#[test]
fn conv_transpose3d_bias_backward_forwards_to_backend() {
    assert_bias_forwarded(conv_transpose3d, |op| {
        matches!(op, Module(ConvTranspose3dBiasBackward(_)))
    });
}

fn assert_bias_forwarded<const D: usize, const N: usize>(
    conv: impl FnOnce(Tensor<D>, Tensor<D>, Option<Tensor<1>>, ConvTransposeOptions<N>) -> Tensor<D>,
    is_expected_op: impl Fn(&OperationIr) -> bool,
) {
    let device = Device::default();
    if !matches!(device.as_dispatch(), burn_dispatch::DispatchDevice::Cube(_)) {
        return;
    }
    let device = device.autodiff();
    let stream = test_stream();
    stream.executes(|| {
        let bias = Tensor::zeros([1], &device).require_grad();
        let mut input_shape = [1; D];
        input_shape[0] = 2;
        let output = conv(
            Tensor::ones(input_shape, &device),
            Tensor::ones([1; D], &device),
            Some(bias.clone()),
            ConvTransposeOptions::new([1; N], [0; N], [0; N], [1; N], 1),
        );
        device.sync().unwrap();

        // Only the bias tracks gradients, so it is the only module operation in backward.
        let inspector = FusionInspector::install(stream);
        let grad = bias.grad(&output.backward()).unwrap();
        grad.into_data().assert_eq(&TensorData::from([2.]), false);

        let reports = inspector.drain();
        assert!(
            reports
                .iter()
                .flat_map(|report| &report.blocks)
                .flat_map(|block| &block.operations)
                .any(is_expected_op),
            "expected a forwarded {N}D bias backward operation, got {reports:#?}"
        );
    });
}
