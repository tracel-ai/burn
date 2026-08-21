use super::*;
use burn_tensor::module::adaptive_avg_pool3d;
use burn_tensor::{Shape, Tolerance};

#[test]
fn test_adaptive_avg_pool3d_backward_simple() {
    let test = AdaptiveAvgPool3dTestCase {
        batch_size: 1,
        channels: 1,
        depth: 2,
        height: 2,
        width: 2,
        depth_out: 1,
        height_out: 1,
        width_out: 1,
    };

    // All 8 elements contribute equally to the single output
    test.assert_output(TestTensor::from_data(
        [[[
            [[0.125, 0.125], [0.125, 0.125]],
            [[0.125, 0.125], [0.125, 0.125]],
        ]]],
        &AutodiffDevice::new(),
    ));
}

#[test]
fn test_adaptive_avg_pool3d_backward_multi_channel() {
    let test = AdaptiveAvgPool3dTestCase {
        batch_size: 1,
        channels: 2,
        depth: 2,
        height: 2,
        width: 2,
        depth_out: 1,
        height_out: 1,
        width_out: 1,
    };

    // Each channel gets equal contribution from its 8 elements
    test.assert_output(TestTensor::from_data(
        [[
            [
                [[0.125, 0.125], [0.125, 0.125]],
                [[0.125, 0.125], [0.125, 0.125]],
            ],
            [
                [[0.125, 0.125], [0.125, 0.125]],
                [[0.125, 0.125], [0.125, 0.125]],
            ],
        ]],
        &AutodiffDevice::new(),
    ));
}

#[test]
fn test_adaptive_avg_pool3d_backward_output_1() {
    // 4x4x4 -> 1x1x1: all 64 elements contribute equally
    let test = AdaptiveAvgPool3dTestCase {
        batch_size: 1,
        channels: 1,
        depth: 4,
        height: 4,
        width: 4,
        depth_out: 1,
        height_out: 1,
        width_out: 1,
    };

    let expected_grad = 1.0 / 64.0;
    test.assert_output(TestTensor::from_data(
        [[[[[expected_grad; 4]; 4]; 4]]],
        &AutodiffDevice::new(),
    ));
}

#[test]
fn test_adaptive_avg_pool3d_backward_dyn_filter() {
    let test = AdaptiveAvgPool3dTestCase {
        batch_size: 1,
        channels: 1,
        depth: 2,
        height: 3,
        width: 4,
        depth_out: 1,
        height_out: 2,
        width_out: 3,
    };

    test.assert_output(TestTensor::from_data(
        [[[
            [
                [0.125, 0.25, 0.25, 0.125],
                [0.25, 0.5, 0.5, 0.25],
                [0.125, 0.25, 0.25, 0.125],
            ],
            [
                [0.125, 0.25, 0.25, 0.125],
                [0.25, 0.5, 0.5, 0.25],
                [0.125, 0.25, 0.25, 0.125],
            ],
        ]]],
        &AutodiffDevice::new(),
    ));
}

#[test]
fn test_adaptive_avg_pool3d_backward_divisible_dyn_filter() {
    let test = AdaptiveAvgPool3dTestCase {
        batch_size: 1,
        channels: 1,
        depth: 4,
        height: 6,
        width: 8,
        depth_out: 2,
        height_out: 3,
        width_out: 4,
    };

    test.assert_output(TestTensor::from_data(
        [[[[[0.125; 8]; 6]; 4]]],
        &AutodiffDevice::new(),
    ));
}

struct AdaptiveAvgPool3dTestCase {
    batch_size: usize,
    channels: usize,
    depth: usize,
    height: usize,
    width: usize,
    depth_out: usize,
    height_out: usize,
    width_out: usize,
}

impl AdaptiveAvgPool3dTestCase {
    fn assert_output(self, x_grad: TestTensor<5>) {
        let shape_x = Shape::new([
            self.batch_size,
            self.channels,
            self.depth,
            self.height,
            self.width,
        ]);
        let device = AutodiffDevice::new();
        let x = TestTensor::from_data(
            TestTensorInt::arange(0..shape_x.num_elements() as i64, &device)
                .reshape::<5, _>(shape_x)
                .into_data(),
            &device,
        )
        .require_grad();
        let output =
            adaptive_avg_pool3d(x.clone(), [self.depth_out, self.height_out, self.width_out]);
        let grads = output.backward();
        let x_grad_actual = x.grad(&grads).unwrap();

        x_grad.to_data().assert_approx_eq::<FloatElem>(
            &x_grad_actual.into_data(),
            Tolerance::default().set_half_precision_relative(1e-3),
        );
    }
}
