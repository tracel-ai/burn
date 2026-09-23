use super::*;
use burn_tensor::Tolerance;
use burn_tensor::module::max_pool3d;

#[test]
fn test_max_pool3d_gradient_single_winner() {
    let device = AutodiffDevice::new();
    let x = TestTensor::<5>::from_data(
        [[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 10.0]]]]],
        &device,
    )
    .require_grad();

    let output = max_pool3d(x.clone(), [2, 2, 2], [1, 1, 1], [0, 0, 0], [1, 1, 1], false);
    let grads = output.backward();

    let x_grad = x.grad(&grads).unwrap();
    // Only the maximum element at (1, 1, 1) receives the gradient 1.0
    let expected = TestTensor::<5>::from_data(
        [[[[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]]]],
        &device,
    );

    expected
        .to_data()
        .assert_approx_eq::<FloatElem>(&x_grad.into_data(), Tolerance::default());
}

#[test]
fn test_max_pool3d_gradient_overlapping_windows() {
    let device = AutodiffDevice::new();
    // 1x1x3x3x3 tensor
    // Put a large value in the center (1, 1, 1) which will win in all overlapping 2x2x2 windows
    let mut data = vec![0.0f32; 27];
    data[1 * 9 + 1 * 3 + 1] = 100.0; // center element

    let x =
        TestTensor::<5>::from_data(burn_tensor::TensorData::new(data, [1, 1, 3, 3, 3]), &device)
            .require_grad();

    // Kernel 2x2x2 with stride 1x1x1 produces 2x2x2 = 8 output windows.
    // The center element (1, 1, 1) is present in ALL 8 windows:
    // d in {0, 1}, h in {0, 1}, w in {0, 1}.
    // Since 100.0 > 0.0, it wins in all 8 windows!
    let output = max_pool3d(x.clone(), [2, 2, 2], [1, 1, 1], [0, 0, 0], [1, 1, 1], false);
    let grads = output.sum().backward();

    let x_grad = x.grad(&grads).unwrap();
    let mut expected_data = vec![0.0f32; 27];
    expected_data[1 * 9 + 1 * 3 + 1] = 8.0;

    let expected = TestTensor::<5>::from_data(
        burn_tensor::TensorData::new(expected_data, [1, 1, 3, 3, 3]),
        &device,
    );

    expected
        .to_data()
        .assert_approx_eq::<FloatElem>(&x_grad.into_data(), Tolerance::default());
}
