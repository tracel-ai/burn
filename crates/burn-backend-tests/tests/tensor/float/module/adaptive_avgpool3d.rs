use super::*;
use burn_tensor::Shape;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;
use burn_tensor::module::adaptive_avg_pool3d;

#[test]
fn test_adaptive_avg_pool3d_simple() {
    let test = AdaptiveAvgPool3dTestCase {
        batch_size: 1,
        channels: 2,
        depth: 4,
        height: 4,
        width: 4,
        depth_out: 2,
        height_out: 2,
        width_out: 2,
    };

    test.assert_output(TestTensor::from([[
        [[[10.5, 12.5], [18.5, 20.5]], [[42.5, 44.5], [50.5, 52.5]]],
        [
            [[74.5, 76.5], [82.5, 84.5]],
            [[106.5, 108.5], [114.5, 116.5]],
        ],
    ]]));
}

#[test]
fn test_adaptive_avg_pool3d_output_1() {
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

    // All 8 elements (0..8) average to 3.5
    test.assert_output(TestTensor::from([[[[[3.5]]]]]));
}

#[test]
fn test_adaptive_avg_pool3d_asymmetric() {
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

    test.assert_output(TestTensor::from([[[[
        [8.5, 9.5, 10.5],
        [12.5, 13.5, 14.5],
    ]]]]));
}

#[test]
fn test_adaptive_avg_pool3d_multichannel_divisible_asymmetric() {
    let test = AdaptiveAvgPool3dTestCase {
        batch_size: 1,
        channels: 2,
        depth: 4,
        height: 6,
        width: 8,
        depth_out: 2,
        height_out: 3,
        width_out: 2,
    };

    test.assert_output(TestTensor::from([[
        [
            [[29.5, 33.5], [45.5, 49.5], [61.5, 65.5]],
            [[125.5, 129.5], [141.5, 145.5], [157.5, 161.5]],
        ],
        [
            [[221.5, 225.5], [237.5, 241.5], [253.5, 257.5]],
            [[317.5, 321.5], [333.5, 337.5], [349.5, 353.5]],
        ],
    ]]));
}

#[test]
fn test_adaptive_avg_pool3d_dyn_filter_size() {
    // Compute expected: for each output position, average the corresponding input window
    let shape_x = Shape::new([1, 2, 5, 7, 9]);
    let x_data: Vec<f32> = (0..shape_x.num_elements() as i64)
        .map(|v| v as f32)
        .collect();
    let device = Default::default();
    let x = TestTensor::from_data(TensorData::new(x_data, shape_x.clone()), &device);
    let output = adaptive_avg_pool3d(x, [3, 4, 5]);
    let output_data = output.into_data();
    assert_eq!(output_data.shape, Shape::new([1, 2, 3, 4, 5]));
    let values: Vec<f32> = output_data.iter::<f32>().collect();

    // Verify all values are finite and positive
    for v in &values {
        assert!(v.is_finite(), "Expected finite, got {}", v);
        assert!(*v >= 0.0, "Expected non-negative, got {}", v);
    }

    // Verify mean is reasonable
    let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
    assert!(mean > 0.0, "Mean should be positive, got {}", mean);
}

#[test]
fn test_adaptive_avg_pool3d_bigger_output() {
    let shape_x = Shape::new([1, 1, 2, 3, 4]);
    let x_data: Vec<f32> = (0..shape_x.num_elements() as i64)
        .map(|v| v as f32)
        .collect();
    let device = Default::default();
    let x = TestTensor::from_data(TensorData::new(x_data, shape_x.clone()), &device);
    let output = adaptive_avg_pool3d(x, [3, 4, 5]);
    let output_data = output.into_data();
    assert_eq!(output_data.shape, Shape::new([1, 1, 3, 4, 5]));
    let values: Vec<f32> = output_data.iter::<f32>().collect();

    for v in &values {
        assert!(v.is_finite(), "Expected finite, got {}", v);
        assert!(*v >= 0.0, "Expected non-negative, got {}", v);
    }

    let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
    // Input is 0..24, mean is 11.5
    assert!(
        (mean - 11.5).abs() < 2.0,
        "Mean {} too far from expected ~11.5",
        mean
    );
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
    fn assert_output(self, y: TestTensor<5>) {
        let shape_x = Shape::new([
            self.batch_size,
            self.channels,
            self.depth,
            self.height,
            self.width,
        ]);
        let x = TestTensor::from(
            TestTensorInt::arange(0..shape_x.num_elements() as i64, &y.device())
                .reshape::<5, _>(shape_x)
                .into_data(),
        );
        let output = adaptive_avg_pool3d(x, [self.depth_out, self.height_out, self.width_out]);

        y.to_data()
            .assert_approx_eq::<FloatElem>(&output.into_data(), Tolerance::default());
    }
}
