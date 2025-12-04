use super::*;
use burn_tensor::Shape;
use burn_tensor::Tolerance;
use burn_tensor::module::avg_pool1d;

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

    test.assert_output(TestTensor::from([[[1., 2., 3., 4.]]]));
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

    test.assert_output(TestTensor::from([[
        [0.33333, 2.0000, 4.0000],
        [4.33333, 8.0000, 10.0000],
    ]]));
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

    test.assert_output(TestTensor::from([[
        [0.5000, 2.0000, 4.0000],
        [6.5000, 8.0000, 10.0000],
    ]]));
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
    fn assert_output(self, y: TestTensor<3>) {
        let shape_x = Shape::new([self.batch_size, self.channels, self.length]);
        let x = TestTensor::from(
            TestTensorInt::arange(0..shape_x.num_elements() as i64, &y.device())
                .reshape::<3, _>(shape_x)
                .into_data(),
        );
        let output = avg_pool1d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.count_include_pad,
        );

        y.to_data().assert_approx_eq::<FloatElem>(
            &output.into_data(),
            Tolerance::default().set_half_precision_relative(1e-3),
        );
    }
}
