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
