use super::*;
use burn_tensor::module::avg_pool2d;
use burn_tensor::{Shape, Tolerance};

#[test]
fn test_avg_pool2d_simple() {
    let test = AvgPool2dTestCase {
        batch_size: 1,
        channels: 1,
        kernel_size_1: 3,
        kernel_size_2: 3,
        padding_1: 0,
        padding_2: 0,
        stride_1: 1,
        stride_2: 1,
        height: 6,
        width: 6,
        count_include_pad: true,
    };

    test.assert_output(TestTensor::from_floats(
        [[[
            [0.11111, 0.22222, 0.33333, 0.33333, 0.22222, 0.11111],
            [0.22222, 0.44444, 0.66667, 0.66667, 0.44444, 0.22222],
            [0.33333, 0.66667, 1.00000, 1.00000, 0.66667, 0.33333],
            [0.33333, 0.66667, 1.00000, 1.00000, 0.66667, 0.33333],
            [0.22222, 0.44444, 0.66667, 0.66667, 0.44444, 0.22222],
            [0.11111, 0.22222, 0.33333, 0.33333, 0.22222, 0.11111],
        ]]],
        &Default::default(),
    ));
}

#[test]
fn test_avg_pool2d_complex() {
    let test = AvgPool2dTestCase {
        batch_size: 1,
        channels: 1,
        kernel_size_1: 3,
        kernel_size_2: 4,
        padding_1: 1,
        padding_2: 2,
        stride_1: 1,
        stride_2: 2,
        height: 4,
        width: 6,
        count_include_pad: true,
    };

    test.assert_output(TestTensor::from_floats(
        [[[
            [0.33333, 0.33333, 0.33333, 0.33333, 0.33333, 0.33333],
            [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
            [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
            [0.33333, 0.33333, 0.33333, 0.33333, 0.33333, 0.33333],
        ]]],
        &Default::default(),
    ));
}

#[test]
fn test_avg_pool2d_complex_dont_include_pad() {
    let test = AvgPool2dTestCase {
        batch_size: 1,
        channels: 1,
        kernel_size_1: 3,
        kernel_size_2: 4,
        padding_1: 1,
        padding_2: 2,
        stride_1: 1,
        stride_2: 2,
        height: 4,
        width: 6,
        count_include_pad: false,
    };

    test.assert_output(TestTensor::from_floats(
        [[[
            [0.6250, 0.6250, 0.41667, 0.41667, 0.6250, 0.6250],
            [0.8750, 0.8750, 0.58333, 0.58333, 0.8750, 0.8750],
            [0.8750, 0.8750, 0.58333, 0.58333, 0.8750, 0.8750],
            [0.6250, 0.6250, 0.41667, 0.41667, 0.6250, 0.6250],
        ]]],
        &Default::default(),
    ));
}

struct AvgPool2dTestCase {
    batch_size: usize,
    channels: usize,
    kernel_size_1: usize,
    kernel_size_2: usize,
    padding_1: usize,
    padding_2: usize,
    stride_1: usize,
    stride_2: usize,
    height: usize,
    width: usize,
    count_include_pad: bool,
}

impl AvgPool2dTestCase {
    fn assert_output(self, x_grad: TestTensor<4>) {
        let shape_x = Shape::new([self.batch_size, self.channels, self.height, self.width]);
        let device = Default::default();
        let x = TestAutodiffTensor::from_data(
            TestTensorInt::arange(0..shape_x.num_elements() as i64, &device)
                .reshape::<4, _>(shape_x)
                .into_data(),
            &device,
        )
        .require_grad();
        let output = avg_pool2d(
            x.clone(),
            [self.kernel_size_1, self.kernel_size_2],
            [self.stride_1, self.stride_2],
            [self.padding_1, self.padding_2],
            self.count_include_pad,
            false,
        );
        let grads = output.backward();
        let x_grad_actual = x.grad(&grads).unwrap();

        x_grad.to_data().assert_approx_eq::<FloatElem>(
            &x_grad_actual.into_data(),
            Tolerance::default().set_half_precision_relative(1e-3),
        );
    }
}
