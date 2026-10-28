use super::*;
use burn_tensor::module::avg_pool3d;
use burn_tensor::ops::AvgPoolOptions;
use burn_tensor::{TensorData, Tolerance};

#[test]
fn test_avg_pool3d_simple_gradient() {
    let device = AutodiffDevice::new();
    let x = TestTensor::<5>::ones([1, 1, 2, 2, 2], &device).require_grad();

    let output = avg_pool3d(
        x.clone(),
        AvgPoolOptions::new([2, 2, 2])
            .with_stride([1, 1, 1])
            .with_padding([0, 0, 0])
            .with_count_include_pad(true)
            .with_ceil_mode(false),
    );
    let grads = output.sum().backward();

    let x_grad = x.grad(&grads).unwrap();
    let expected = TestTensor::<5>::from_data(
        [[[
            [[0.125, 0.125], [0.125, 0.125]],
            [[0.125, 0.125], [0.125, 0.125]],
        ]]],
        &device,
    );

    expected
        .to_data()
        .assert_approx_eq::<FloatElem>(&x_grad.into_data(), Tolerance::default());
}

#[test]
fn test_avg_pool3d_count_exclude_pad_gradient() {
    let device = AutodiffDevice::new();
    let x = TestTensor::<5>::ones([1, 1, 2, 2, 2], &device).require_grad();

    // Pool with kernel [2, 2, 2], stride [2, 2, 2], padding [1, 1, 1], count_include_pad=false
    // Output is 2x2x2. Each output window covers exactly 1 unpadded element from x.
    // So divisor is 1, gradient from that window to that single element is 1.0.
    let output = avg_pool3d(
        x.clone(),
        AvgPoolOptions::new([2, 2, 2])
            .with_stride([2, 2, 2])
            .with_padding([1, 1, 1])
            .with_count_include_pad(false)
            .with_ceil_mode(false),
    );
    let grads = output.sum().backward();

    let x_grad = x.grad(&grads).unwrap();
    let expected = TestTensor::<5>::ones([1, 1, 2, 2, 2], &device);

    expected
        .to_data()
        .assert_approx_eq::<FloatElem>(&x_grad.into_data(), Tolerance::default());
}

#[test]
fn test_avg_pool3d_dilation_gradient() {
    // Kernel [2, 1, 2], dilation [2, 1, 2], stride 1, padding [1, 0, 1] over a 3x2x3 input.
    let device = AutodiffDevice::new();
    for (count_include_pad, expected) in [
        (
            true,
            [
                [[2.0, 4.0, 2.0], [2.75, 5.5, 2.75]],
                [[4.0, 8.0, 4.0], [5.5, 11.0, 5.5]],
                [[2.0, 4.0, 2.0], [2.75, 5.5, 2.75]],
            ],
        ),
        (
            false,
            [
                [[2.0, 8.0, 2.0], [2.75, 11.0, 2.75]],
                [[8.0, 32.0, 8.0], [11.0, 44.0, 11.0]],
                [[2.0, 8.0, 2.0], [2.75, 11.0, 2.75]],
            ],
        ),
    ] {
        let x = TestTensor::<5>::ones([1, 1, 3, 2, 3], &device).require_grad();
        let output = avg_pool3d(
            x.clone(),
            AvgPoolOptions::new([2, 1, 2])
                .with_stride([1, 1, 1])
                .with_padding([1, 0, 1])
                .with_dilation([2, 1, 2])
                .with_count_include_pad(count_include_pad),
        );
        let [_, _, d, h, w] = output.dims();
        let upstream = TestTensor::from_data(
            TensorData::new((1..=d * h * w).map(|v| v as f32).collect(), output.shape()),
            &device,
        );
        let grads = (output * upstream).backward();
        x.grad(&grads)
            .unwrap()
            .into_data()
            .assert_approx_eq::<FloatElem>(&TensorData::from([[expected]]), Tolerance::default());
    }
}
