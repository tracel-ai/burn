use super::*;
use burn_tensor::Tolerance;
use burn_tensor::module::avg_pool3d;

#[test]
fn test_avg_pool3d_simple_gradient() {
    let device = AutodiffDevice::new();
    let x = TestTensor::<5>::ones([1, 1, 2, 2, 2], &device).require_grad();

    let output = avg_pool3d(x.clone(), [2, 2, 2], [1, 1, 1], [0, 0, 0], true, false);
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
    let output = avg_pool3d(x.clone(), [2, 2, 2], [2, 2, 2], [1, 1, 1], false, false);
    let grads = output.sum().backward();

    let x_grad = x.grad(&grads).unwrap();
    let expected = TestTensor::<5>::ones([1, 1, 2, 2, 2], &device);

    expected
        .to_data()
        .assert_approx_eq::<FloatElem>(&x_grad.into_data(), Tolerance::default());
}
