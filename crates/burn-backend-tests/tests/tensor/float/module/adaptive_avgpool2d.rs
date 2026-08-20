use super::*;
use burn_tensor::Shape;
use burn_tensor::Tolerance;
use burn_tensor::module::adaptive_avg_pool2d;

#[test]
fn test_adaptive_avg_pool2d_simple() {
    let test = AdaptiveAvgPool2dTestCase {
        batch_size: 1,
        channels: 2,
        height: 8,
        width: 6,
        height_out: 4,
        width_out: 4,
    };

    test.assert_output(TestTensor::from([[
        [
            [3.5000, 4.5000, 6.5000, 7.5000],
            [15.5000, 16.5000, 18.5000, 19.5000],
            [27.5000, 28.5000, 30.5000, 31.5000],
            [39.5000, 40.5000, 42.5000, 43.5000],
        ],
        [
            [51.5000, 52.5000, 54.5000, 55.5000],
            [63.5000, 64.5000, 66.5000, 67.5000],
            [75.5000, 76.5000, 78.5000, 79.5000],
            [87.5000, 88.5000, 90.5000, 91.5000],
        ],
    ]]));
}

#[test]
fn test_adaptive_avg_pool2d_dyn_filter_size() {
    let test = AdaptiveAvgPool2dTestCase {
        batch_size: 1,
        channels: 2,
        height: 5,
        width: 7,
        height_out: 3,
        width_out: 2,
    };

    test.assert_output(TestTensor::from([[
        [[5.0000, 8.0000], [15.5000, 18.5000], [26.0000, 29.0000]],
        [[40.0000, 43.0000], [50.5000, 53.5000], [61.0000, 64.0000]],
    ]]));
}

#[test]
fn test_adaptive_avg_pool2d_bigger_output() {
    let test = AdaptiveAvgPool2dTestCase {
        batch_size: 1,
        channels: 2,
        height: 4,
        width: 3,
        height_out: 5,
        width_out: 4,
    };

    test.assert_output(TestTensor::from([[
        [
            [0.0000, 0.5000, 1.5000, 2.0000],
            [1.5000, 2.0000, 3.0000, 3.5000],
            [4.5000, 5.0000, 6.0000, 6.5000],
            [7.5000, 8.0000, 9.0000, 9.5000],
            [9.0000, 9.5000, 10.5000, 11.0000],
        ],
        [
            [12.0000, 12.5000, 13.5000, 14.0000],
            [13.5000, 14.0000, 15.0000, 15.5000],
            [16.5000, 17.0000, 18.0000, 18.5000],
            [19.5000, 20.0000, 21.0000, 21.5000],
            [21.0000, 21.5000, 22.5000, 23.0000],
        ],
    ]]));
}

/// A `1x1` output takes a dedicated path on the cubecl backends: it is a
/// reduction over every pixel rather than a pooling window, and is dispatched to
/// the reduce kernels because the pooling one parallelises over its output and
/// would run this with `batch * channels` units. Nothing else here covers a
/// `1x1` output, so nothing else covers that path.
#[test]
fn test_adaptive_avg_pool2d_global() {
    let test = AdaptiveAvgPool2dTestCase {
        batch_size: 2,
        channels: 2,
        height: 4,
        width: 5,
        height_out: 1,
        width_out: 1,
    };

    // Each channel is 20 consecutive integers, so its mean is its midpoint.
    test.assert_output(TestTensor::from([
        [[[9.5000]], [[29.5000]]],
        [[[49.5000]], [[69.5000]]],
    ]));
}

/// The same path over a map whose pixel count is neither square nor a power of
/// two, so a reduction that padded or rounded its axis would show up here.
#[test]
fn test_adaptive_avg_pool2d_global_odd_shape() {
    let test = AdaptiveAvgPool2dTestCase {
        batch_size: 1,
        channels: 2,
        height: 5,
        width: 7,
        height_out: 1,
        width_out: 1,
    };

    test.assert_output(TestTensor::from([[[[17.0000]], [[52.0000]]]]));
}

struct AdaptiveAvgPool2dTestCase {
    batch_size: usize,
    channels: usize,
    height: usize,
    width: usize,
    height_out: usize,
    width_out: usize,
}

impl AdaptiveAvgPool2dTestCase {
    fn assert_output(self, y: TestTensor<4>) {
        let shape_x = Shape::new([self.batch_size, self.channels, self.height, self.width]);
        let x = TestTensor::from(
            TestTensorInt::arange(0..shape_x.num_elements() as i64, &y.device())
                .reshape::<4, _>(shape_x)
                .into_data(),
        );
        let output = adaptive_avg_pool2d(x, [self.height_out, self.width_out]);

        y.to_data()
            .assert_approx_eq::<FloatElem>(&output.into_data(), Tolerance::default());
    }
}
