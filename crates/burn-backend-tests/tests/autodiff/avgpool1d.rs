use super::*;
use burn_tensor::module::avg_pool1d;
use burn_tensor::{Shape, TensorData, Tolerance};

#[test]
fn test_avg_pool1d_backward_writes_every_channel() {
    let device = AutodiffDevice::new();
    let poison =
        TestTensor::<2>::from_data(TensorData::new(vec![1234.5; 8192], [64, 128]), &device);
    let _ = (poison.clone() * poison).sum().into_data();

    for channels in [2, 4] {
        let x = TestTensor::<3>::from_data(
            TensorData::new(
                (0..channels * 6)
                    .map(|index| index as f32 * 0.1 + 0.5)
                    .collect(),
                [1, channels, 6],
            ),
            &device,
        )
        .require_grad();
        let output = avg_pool1d(x.clone(), 3, 2, 1, true, false);
        let grads = output.sum().backward();
        let actual = x.grad(&grads).unwrap();
        let expected = TestTensor::<3>::from_data(
            TensorData::new(
                vec![
                    1.0 / 3.0,
                    2.0 / 3.0,
                    1.0 / 3.0,
                    2.0 / 3.0,
                    1.0 / 3.0,
                    1.0 / 3.0,
                ]
                .into_iter()
                .cycle()
                .take(channels * 6)
                .collect(),
                [1, channels, 6],
            ),
            &device,
        );

        expected
            .to_data()
            .assert_approx_eq::<FloatElem>(&actual.into_data(), Tolerance::default());
    }
}

#[test]
fn test_avg_pool1d_simple() {
    let test = AvgPool1dTestCase {
        batch_size: 1,
        channels: 1,
        kernel_size: 3,
        padding: 0,
        stride: 1,
        length: 6,
        count_include_pad: true,
    };

    test.assert_output(TestTensor::from_data(
        [[[0.33333, 0.66667, 1.0000, 1.0000, 0.66667, 0.33333]]],
        &AutodiffDevice::new(),
    ));
}

#[test]
fn test_avg_pool1d_complex() {
    let test = AvgPool1dTestCase {
        batch_size: 1,
        channels: 2,
        kernel_size: 3,
        padding: 1,
        stride: 2,
        length: 6,
        count_include_pad: true,
    };

    test.assert_output(TestTensor::from_data(
        [[
            [0.33333, 0.66667, 0.33333, 0.66667, 0.33333, 0.33333],
            [0.33333, 0.66667, 0.33333, 0.66667, 0.33333, 0.33333],
        ]],
        &AutodiffDevice::new(),
    ));
}

#[test]
fn test_avg_pool1d_complex_dont_count_pad() {
    let test = AvgPool1dTestCase {
        batch_size: 1,
        channels: 2,
        kernel_size: 3,
        padding: 1,
        stride: 2,
        length: 6,
        count_include_pad: false,
    };

    test.assert_output(TestTensor::from_data(
        [[
            [0.5000, 0.83333, 0.33333, 0.66667, 0.33333, 0.33333],
            [0.5000, 0.83333, 0.33333, 0.66667, 0.33333, 0.33333],
        ]],
        &AutodiffDevice::new(),
    ));
}

struct AvgPool1dTestCase {
    batch_size: usize,
    channels: usize,
    kernel_size: usize,
    padding: usize,
    stride: usize,
    length: usize,
    count_include_pad: bool,
}

impl AvgPool1dTestCase {
    fn assert_output(self, x_grad: TestTensor<3>) {
        let shape_x = Shape::new([self.batch_size, self.channels, self.length]);
        let device = AutodiffDevice::new();
        let x = TestTensor::from_data(
            TestTensorInt::arange(0..shape_x.num_elements() as i64, &device)
                .reshape::<3, _>(shape_x)
                .into_data(),
            &device,
        )
        .require_grad();
        let output = avg_pool1d(
            x.clone(),
            self.kernel_size,
            self.stride,
            self.padding,
            self.count_include_pad,
            false,
        );
        let grads = output.backward();
        let x_grad_actual = x.grad(&grads).unwrap();

        let tolerance = Tolerance::default().set_half_precision_relative(1e-3);
        x_grad
            .to_data()
            .assert_approx_eq::<FloatElem>(&x_grad_actual.into_data(), tolerance);
    }
}
