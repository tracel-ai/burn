use super::*;
use burn_tensor::{
    DType, Device, TensorData,
    module::{conv_transpose1d, interpolate, max_pool2d},
    ops::{ConvTransposeOptions, InterpolateMode, InterpolateOptions},
};

fn add_zero_and_max_pool2d(dev: Device, dtype: DType) -> TensorData {
    let input = TestTensor::<4>::from_data(
        TensorData::from([
            [[[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]],
            [[[2.0, 2.0], [2.0, 2.0]], [[3.0, 3.0], [3.0, 3.0]]],
        ]),
        &dev,
    )
    .cast(dtype);

    let zeros = TestTensor::<4>::zeros(input.shape(), &dev).cast(dtype);

    dev.sync().unwrap();

    // Elementwise operation (add a test tensor with 0 values)
    let x = input + zeros;

    // Max pool 2d
    let output = max_pool2d(
        x,
        [2, 2], // kernel_size
        [2, 2], // stride
        [0, 0], // padding
        [0, 0], // dilation
        false,  // ceil_mode
    );

    output.into_data()
}

#[test]
fn fusion_test_elementwise_operation_followed_by_max_pool2d() {
    let fused_32 = add_zero_and_max_pool2d(Default::default(), DType::F32);

    assert_eq!(fused_32.shape.to_vec(), vec![2, 2, 1, 1]);
    let expected_values: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0];
    let actual_values = fused_32.to_vec::<f32>().unwrap();

    assert_eq!(actual_values, expected_values);
}

fn add_zero_and_interpolate_nearest(dev: Device, dtype: DType) -> TensorData {
    let input = TestTensor::<4>::from_data(
        TensorData::from([
            [[[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]],
            [[[2.0, 2.0], [2.0, 2.0]], [[3.0, 3.0], [3.0, 3.0]]],
        ]),
        &dev,
    )
    .cast(dtype);

    let zeros = TestTensor::<4>::zeros(input.shape(), &dev).cast(dtype);

    dev.sync().unwrap();

    // Elementwise operation (add a test tensor with 0 values)
    let x = input + zeros;

    // Max pool 2d
    let output = interpolate(
        x,
        [4, 4],
        InterpolateOptions {
            mode: InterpolateMode::Nearest,
            align_corners: false,
        },
    );

    output.into_data()
}

#[test]
fn fusion_test_elementwise_operation_followed_by_interpolate_nearest() {
    let fused_32 = add_zero_and_interpolate_nearest(Default::default(), DType::F32);
    // The shape is scaled from [2, 2, 2, 2] to [2, 2, 4, 4]
    assert_eq!(fused_32.shape.to_vec(), vec![2, 2, 4, 4]);

    // Since each channel originally contained uniform 2x2 values,
    // the 4x4 interpolated output will contain 16 of each value.
    let mut expected_values: Vec<f32> = Vec::with_capacity(64);
    expected_values.extend(vec![0.0; 16]); // N=0, C=0
    expected_values.extend(vec![1.0; 16]); // N=0, C=1
    expected_values.extend(vec![2.0; 16]); // N=1, C=0
    expected_values.extend(vec![3.0; 16]); // N=1, C=1

    let actual_values = fused_32.to_vec::<f32>().unwrap();
    assert_eq!(actual_values, expected_values);
}

fn add_zero_and_conv_transpose1d(dev: Device, dtype: DType) -> TensorData {
    let input =
        TestTensor::<3>::from_data(TensorData::from([[[0.0, 1.0, 2.0, 3.0]]]), &dev).cast(dtype);
    let weight = TestTensor::<3>::from_data(TensorData::from([[[2.0, 3.0]]]), &dev).cast(dtype);

    let input_zeros = TestTensor::<3>::zeros(input.shape(), &dev).cast(dtype);
    let weight_zeros = TestTensor::<3>::zeros(weight.shape(), &dev).cast(dtype);

    dev.sync().unwrap();

    let x = input + input_zeros;
    let weight = weight + weight_zeros;

    let output = conv_transpose1d(
        x,
        weight,
        None,
        ConvTransposeOptions::new([1], [0], [0], [1], 1),
    );

    output.into_data()
}

#[test]
fn fusion_test_elementwise_operation_followed_by_conv_transpose1d() {
    let fused_32 = add_zero_and_conv_transpose1d(Default::default(), DType::F32);

    assert_eq!(fused_32.shape.to_vec(), vec![1, 1, 4]);
    let expected_values: Vec<f32> = vec![0.0, 2.0, 4.0, 6.0];
    let actual_values = fused_32.to_vec::<f32>().unwrap();

    assert_eq!(actual_values, expected_values);
}
