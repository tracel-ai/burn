use super::*;
use burn_tensor::module::adaptive_avg_pool2d;
use burn_tensor::{Shape, Tolerance};

#[test]
fn test_avg_pool2d_simple() {
    let test = AdaptiveAvgPool2dTestCase {
        batch_size: 1,
        channels: 2,
        height: 5,
        width: 3,
        output_size_1: 3,
        output_size_2: 2,
    };

    test.assert_output(TestTensor::from_floats(
        [[
            [
                [0.2500, 0.5000, 0.2500],
                [0.41667, 0.83333, 0.41667],
                [0.16667, 0.33333, 0.16667],
                [0.41667, 0.83333, 0.41667],
                [0.2500, 0.5000, 0.2500],
            ],
            [
                [0.2500, 0.5000, 0.2500],
                [0.41667, 0.83333, 0.41667],
                [0.16667, 0.33333, 0.16667],
                [0.41667, 0.83333, 0.41667],
                [0.2500, 0.5000, 0.2500],
            ],
        ]],
        &Default::default(),
    ));
}

#[test]
fn test_avg_pool2d_output_1() {
    let test = AdaptiveAvgPool2dTestCase {
        batch_size: 1,
        channels: 1,
        height: 4,
        width: 8,
        output_size_1: 1,
        output_size_2: 1,
    };

    test.assert_output(TestTensor::from_floats(
        [[[
            [
                0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125,
            ],
            [
                0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125,
            ],
            [
                0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125,
            ],
            [
                0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125,
            ],
        ]]],
        &Default::default(),
    ));
}

struct AdaptiveAvgPool2dTestCase {
    batch_size: usize,
    channels: usize,
    height: usize,
    width: usize,
    output_size_1: usize,
    output_size_2: usize,
}

impl AdaptiveAvgPool2dTestCase {
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
        let output = adaptive_avg_pool2d(x.clone(), [self.output_size_1, self.output_size_2]);
        let grads = output.backward();
        let x_grad_actual = x.grad(&grads).unwrap();

        x_grad.to_data().assert_approx_eq::<FloatElem>(
            &x_grad_actual.into_data(),
            Tolerance::default().set_half_precision_relative(1e-3),
        );
    }
}
