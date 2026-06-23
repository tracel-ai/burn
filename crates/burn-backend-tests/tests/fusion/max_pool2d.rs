use super::*;
use burn_tensor::{DType, Device, TensorData, module::max_pool2d};

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
