use super::*;
use burn_tensor::Tolerance;
use burn_tensor::module::max_pool2d;

#[test]
fn test_max_pool2d_simple_1() {
    let kernel_size_1 = 3;
    let kernel_size_2 = 3;
    let padding_1 = 0;
    let padding_2 = 0;
    let stride_1 = 1;
    let stride_2 = 1;
    let dilation_1 = 1;
    let dilation_2 = 1;

    let device = Default::default();
    let x = TestAutodiffTensor::from_floats(
        [[[
            [0.2479, 0.6386, 0.3166, 0.5742],
            [0.7065, 0.1940, 0.6305, 0.8959],
            [0.5416, 0.8602, 0.8129, 0.1662],
            [0.3358, 0.3059, 0.8293, 0.0990],
        ]]],
        &device,
    )
    .require_grad();
    let x_grad_expected = TestAutodiffTensor::<4>::from_floats(
        [[[
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]]],
        &device,
    );

    let output = max_pool2d(
        x.clone(),
        [kernel_size_1, kernel_size_2],
        [stride_1, stride_2],
        [padding_1, padding_2],
        [dilation_1, dilation_2],
        false,
    );
    let grads = output.backward();

    // Asserts
    let x_grad_actual = x.grad(&grads).unwrap();
    x_grad_expected
        .to_data()
        .assert_approx_eq::<FloatElem>(&x_grad_actual.to_data(), Tolerance::default());
}

#[test]
fn test_max_pool2d_simple_2() {
    let kernel_size_1 = 2;
    let kernel_size_2 = 2;
    let padding_1 = 1;
    let padding_2 = 1;
    let stride_1 = 1;
    let stride_2 = 1;
    let dilation_1 = 1;
    let dilation_2 = 1;

    let device = Default::default();
    let x = TestAutodiffTensor::from_floats(
        [[[
            [0.2479, 0.6386, 0.3166, 0.5742],
            [0.7065, 0.1940, 0.6305, 0.8959],
            [0.5416, 0.8602, 0.8129, 0.1662],
            [0.3358, 0.3059, 0.8293, 0.0990],
        ]]],
        &device,
    )
    .require_grad();
    let x_grad_expected = TestAutodiffTensor::<4>::from_floats(
        [[[
            [1., 3., 0., 2.],
            [3., 0., 0., 4.],
            [1., 4., 0., 1.],
            [2., 0., 3., 1.],
        ]]],
        &device,
    );

    let output = max_pool2d(
        x.clone(),
        [kernel_size_1, kernel_size_2],
        [stride_1, stride_2],
        [padding_1, padding_2],
        [dilation_1, dilation_2],
        false,
    );
    let grads = output.backward();

    // Asserts
    let x_grad_actual = x.grad(&grads).unwrap();
    x_grad_expected
        .to_data()
        .assert_approx_eq::<FloatElem>(&x_grad_actual.to_data(), Tolerance::default());
}

#[test]
fn test_max_pool2d_with_dilation() {
    let kernel_size_1 = 2;
    let kernel_size_2 = 2;
    let padding_1 = 1;
    let padding_2 = 1;
    let stride_1 = 1;
    let stride_2 = 1;
    let dilation_1 = 2;
    let dilation_2 = 2;

    let device = Default::default();
    let x = TestAutodiffTensor::from_floats(
        [[[
            [0.2479, 0.6386, 0.3166, 0.5742],
            [0.7065, 0.1940, 0.6305, 0.8959],
            [0.5416, 0.8602, 0.8129, 0.1662],
            [0.3358, 0.3059, 0.8293, 0.0990],
        ]]],
        &device,
    )
    .require_grad();
    let x_grad_expected = TestAutodiffTensor::<4>::from_floats(
        [[[
            [0., 0., 0., 0.],
            [1., 1., 1., 2.],
            [0., 4., 4., 0.],
            [0., 1., 2., 0.],
        ]]],
        &device,
    );

    let output = max_pool2d(
        x.clone(),
        [kernel_size_1, kernel_size_2],
        [stride_1, stride_2],
        [padding_1, padding_2],
        [dilation_1, dilation_2],
        false,
    );
    let grads = output.backward();

    // Asserts
    let x_grad_actual = x.grad(&grads).unwrap();
    x_grad_expected
        .to_data()
        .assert_approx_eq::<FloatElem>(&x_grad_actual.to_data(), Tolerance::default());
}

#[test]
fn test_max_pool2d_complex() {
    let kernel_size_1 = 4;
    let kernel_size_2 = 2;
    let padding_1 = 2;
    let padding_2 = 1;
    let stride_1 = 1;
    let stride_2 = 2;
    let dilation_1 = 1;
    let dilation_2 = 1;

    let device = Default::default();
    let x = TestAutodiffTensor::from_floats(
        [[[
            [0.5388, 0.0676, 0.7122, 0.8316, 0.0653],
            [0.9154, 0.1536, 0.9089, 0.8016, 0.7518],
            [0.2073, 0.0501, 0.8811, 0.5604, 0.5075],
            [0.4384, 0.9963, 0.9698, 0.4988, 0.2609],
            [0.3391, 0.2230, 0.4610, 0.5365, 0.6880],
        ]]],
        &device,
    )
    .require_grad();
    let x_grad_expected = TestAutodiffTensor::<4>::from_floats(
        [[[
            [0., 0., 0., 3., 0.],
            [4., 0., 2., 1., 0.],
            [0., 0., 0., 0., 0.],
            [2., 4., 0., 0., 0.],
            [0., 0., 0., 0., 2.],
        ]]],
        &device,
    );

    let output = max_pool2d(
        x.clone(),
        [kernel_size_1, kernel_size_2],
        [stride_1, stride_2],
        [padding_1, padding_2],
        [dilation_1, dilation_2],
        false,
    );
    let grads = output.backward();

    // Asserts
    let x_grad_actual = x.grad(&grads).unwrap();
    x_grad_expected
        .to_data()
        .assert_approx_eq::<FloatElem>(&x_grad_actual.to_data(), Tolerance::default());
}

#[test]
fn test_max_pool2d_ceil_mode() {
    // Test ceil_mode=true with gradient computation
    // Using 1x1x6x6 input with kernel 3x3, stride 2x2, padding 0
    // Floor mode: output 2x2
    // Ceil mode: output 3x3
    let kernel_size_1 = 3;
    let kernel_size_2 = 3;
    let padding_1 = 0;
    let padding_2 = 0;
    let stride_1 = 2;
    let stride_2 = 2;
    let dilation_1 = 1;
    let dilation_2 = 1;

    let device = Default::default();
    // Input (values 1-36):
    let x = TestAutodiffTensor::from_floats(
        [[[
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
            [19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
            [25.0, 26.0, 27.0, 28.0, 29.0, 30.0],
            [31.0, 32.0, 33.0, 34.0, 35.0, 36.0],
        ]]],
        &device,
    )
    .require_grad();

    // Expected gradients for ceil_mode output 3x3:
    // Output positions and their max value positions:
    // (0,0): max at (2,2)=15 -> grad[2,2] += 1
    // (0,1): max at (2,4)=17 -> grad[2,4] += 1
    // (0,2): max at (2,5)=18 -> grad[2,5] += 1
    // (1,0): max at (4,2)=27 -> grad[4,2] += 1
    // (1,1): max at (4,4)=29 -> grad[4,4] += 1
    // (1,2): max at (4,5)=30 -> grad[4,5] += 1
    // (2,0): max at (5,2)=33 -> grad[5,2] += 1
    // (2,1): max at (5,4)=35 -> grad[5,4] += 1
    // (2,2): max at (5,5)=36 -> grad[5,5] += 1
    let x_grad_expected = TestAutodiffTensor::<4>::from_floats(
        [[[
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 1., 1.],
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 1., 1.],
            [0., 0., 1., 0., 1., 1.],
        ]]],
        &device,
    );

    let output = max_pool2d(
        x.clone(),
        [kernel_size_1, kernel_size_2],
        [stride_1, stride_2],
        [padding_1, padding_2],
        [dilation_1, dilation_2],
        true,
    );
    let grads = output.backward();

    // Asserts
    let x_grad_actual = x.grad(&grads).unwrap();
    x_grad_expected
        .to_data()
        .assert_approx_eq::<FloatElem>(&x_grad_actual.to_data(), Tolerance::default());
}
