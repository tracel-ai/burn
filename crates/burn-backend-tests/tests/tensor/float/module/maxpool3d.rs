use super::*;
use burn_tensor::Tolerance;
use burn_tensor::module::{max_pool3d, max_pool3d_with_indices};

#[test]
fn test_max_pool3d_simple() {
    let kernel_size = [2, 2, 2];
    let stride = [1, 1, 1];
    let padding = [0, 0, 0];
    let dilation = [1, 1, 1];

    // 1x1x3x3x3 tensor with values 0..27
    let shape_x = burn_tensor::Shape::new([1, 1, 3, 3, 3]);
    let x = TestTensor::from(
        TestTensorInt::arange(0..shape_x.num_elements() as i64, &Default::default())
            .reshape::<5, _>(shape_x)
            .into_data(),
    );

    // In each 2x2x2 window, max is at the bottom-right-far corner:
    // Window (0, 0, 0): max is at (1, 1, 1) = 1 * 9 + 1 * 3 + 1 = 13
    // Window (0, 0, 1): max is at (1, 1, 2) = 1 * 9 + 1 * 3 + 2 = 14
    // Window (0, 1, 0): max is at (1, 2, 1) = 1 * 9 + 2 * 3 + 1 = 16
    // Window (0, 1, 1): max is at (1, 2, 2) = 1 * 9 + 2 * 3 + 2 = 17
    // Window (1, 0, 0): max is at (2, 1, 1) = 2 * 9 + 1 * 3 + 1 = 22
    // Window (1, 0, 1): max is at (2, 1, 2) = 2 * 9 + 1 * 3 + 2 = 23
    // Window (1, 1, 0): max is at (2, 2, 1) = 2 * 9 + 2 * 3 + 1 = 25
    // Window (1, 1, 1): max is at (2, 2, 2) = 2 * 9 + 2 * 3 + 2 = 26
    let y_expected =
        TestTensor::<5>::from([[[[[13.0, 14.0], [16.0, 17.0]], [[22.0, 23.0], [25.0, 26.0]]]]]);

    let output = max_pool3d(x, kernel_size, stride, padding, dilation, false);

    y_expected
        .to_data()
        .assert_approx_eq::<FloatElem>(&output.into_data(), Tolerance::default());
}

#[test]
fn test_max_pool3d_with_indices() {
    let kernel_size = [2, 2, 2];
    let stride = [1, 1, 1];
    let padding = [0, 0, 0];
    let dilation = [1, 1, 1];

    let shape_x = burn_tensor::Shape::new([1, 1, 3, 3, 3]);
    let x = TestTensor::from(
        TestTensorInt::arange(0..shape_x.num_elements() as i64, &Default::default())
            .reshape::<5, _>(shape_x)
            .into_data(),
    );

    let (output, indices) =
        max_pool3d_with_indices(x, kernel_size, stride, padding, dilation, false);

    let y_expected =
        TestTensor::<5>::from([[[[[13.0, 14.0], [16.0, 17.0]], [[22.0, 23.0], [25.0, 26.0]]]]]);

    // Flat spatial indices: id * (H * W) + ih * W + iw
    // Here H=3, W=3, H*W=9
    let indices_expected = TestTensorInt::<5>::from_data(
        [[[[[13, 14], [16, 17]], [[22, 23], [25, 26]]]]],
        &Default::default(),
    );

    y_expected
        .to_data()
        .assert_approx_eq::<FloatElem>(&output.into_data(), Tolerance::default());
    assert_eq!(indices.into_data(), indices_expected.into_data());
}

#[test]
fn test_max_pool3d_ceil_mode() {
    let x = TestTensor::<5>::ones([1, 1, 5, 5, 5], &Default::default());

    let out_floor = max_pool3d(x.clone(), [2, 2, 2], [2, 2, 2], [0, 0, 0], [1, 1, 1], false);
    assert_eq!(out_floor.dims(), [1, 1, 2, 2, 2]);

    let out_ceil = max_pool3d(x, [2, 2, 2], [2, 2, 2], [0, 0, 0], [1, 1, 1], true);
    assert_eq!(out_ceil.dims(), [1, 1, 3, 3, 3]);
}

#[test]
fn test_max_pool3d_discard_branch() {
    // 5x5x5 input, kernel 2, stride 2, padding 1, ceil_mode = true
    // Window 3 starting at index 6 >= 5 + 1 = 6 is discarded
    let x = TestTensor::<5>::ones([1, 1, 5, 5, 5], &Default::default());
    let out = max_pool3d(x, [2, 2, 2], [2, 2, 2], [1, 1, 1], [1, 1, 1], true);
    assert_eq!(out.dims(), [1, 1, 3, 3, 3]);
}

#[test]
fn test_max_pool3d_non_arange_multichannel_with_indices() {
    let kernel_size = [2, 2, 2];
    let stride = [2, 2, 2];
    let padding = [1, 1, 1];
    let dilation = [1, 1, 1];

    // Input shape [2, 2, 2, 2, 2] with non-arange, distinct values across batches and channels.
    // For each (b, c) channel cube, spatial extent is [2, 2, 2] (total 8 elements, flat indices 0..7).
    // With kernel 2, stride 2, padding 1: each window (od, oh, ow) in the 2x2x2 output covers exactly
    // one in-bounds input element at (od, oh, ow) and 7 out-of-bounds padding values (-inf).
    // Thus the max in window (od, oh, ow) is precisely the input element at (od, oh, ow)
    // with flat spatial index: od * 4 + oh * 2 + ow.
    let x_data = [
        // Batch 0
        [
            // Channel 0: decreasing positive values
            [[[80.0, 70.0], [60.0, 50.0]], [[40.0, 30.0], [20.0, 10.0]]],
            // Channel 1: all negative values (values < 0 to ensure indices are not conflated with values)
            [
                [[-10.0, -20.0], [-30.0, -40.0]],
                [[-50.0, -60.0], [-70.0, -80.0]],
            ],
        ],
        // Batch 1
        [
            // Channel 0: arbitrary values
            [[[15.5, 3.2], [99.1, 42.0]], [[0.0, 88.8], [7.7, 55.5]]],
            // Channel 1: large values
            [
                [[1000.0, 2000.0], [3000.0, 4000.0]],
                [[5000.0, 6000.0], [7000.0, 8000.0]],
            ],
        ],
    ];

    let x = TestTensor::<5>::from(x_data);
    let (output, indices) =
        max_pool3d_with_indices(x.clone(), kernel_size, stride, padding, dilation, false);

    // Output values should match input values exactly
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&x.into_data(), Tolerance::default());

    // Flat spatial indices: 0..7 for each (batch, channel) cube, isolating spatial coordinates
    let expected_indices = TestTensorInt::<5>::from_data(
        [
            [
                [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
                [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
            ],
            [
                [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
                [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
            ],
        ],
        &Default::default(),
    );
    assert_eq!(indices.into_data(), expected_indices.into_data());
}

#[test]
fn test_max_pool3d_dilation() {
    let kernel_size = [2, 2, 2];
    let stride = [1, 1, 1];
    let padding = [0, 0, 0];
    let dilation = [2, 2, 2];

    // Input shape: [1, 1, 3, 3, 3] (total 27 elements)
    // Effective kernel size: 2 * (2 - 1) + 1 = 3.
    // Output spatial size: (3 - 3) / 1 + 1 = 1.
    // Kernel taps at offsets (k * 2): {0, 2} along d, h, w.
    // Tapped spatial coordinates are:
    // (0,0,0)->0, (0,0,2)->2, (0,2,0)->6, (0,2,2)->8,
    // (2,0,0)->18, (2,0,2)->20, (2,2,0)->24, (2,2,2)->26.
    // Position (1, 1, 1) -> index 13 is SKIPPED by dilation.
    // Place a huge distractor at (1, 1, 1): if dilation is respected, it is ignored!
    let mut data = vec![0.0f32; 27];
    data[13] = 999.0; // center element (1, 1, 1) - skipped by dilation
    data[0] = 10.0;
    data[2] = 20.0;
    data[6] = 30.0;
    data[8] = 40.0;
    data[18] = 50.0;
    data[20] = 60.0;
    data[24] = 70.0;
    data[26] = 80.0; // maximum among tapped elements

    let x = TestTensor::<5>::from_data(
        burn_tensor::TensorData::new(data, [1, 1, 3, 3, 3]),
        &Default::default(),
    );

    let (output, indices) =
        max_pool3d_with_indices(x, kernel_size, stride, padding, dilation, false);

    assert_eq!(output.dims(), [1, 1, 1, 1, 1]);
    let expected_output = TestTensor::<5>::from([[[[[80.0]]]]]);
    expected_output
        .to_data()
        .assert_approx_eq::<FloatElem>(&output.into_data(), Tolerance::default());

    let expected_indices = TestTensorInt::<5>::from_data([[[[[26]]]]], &Default::default());
    assert_eq!(indices.into_data(), expected_indices.into_data());
}
