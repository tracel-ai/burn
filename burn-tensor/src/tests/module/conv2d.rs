#[burn_tensor_testgen::testgen(module_conv2d)]
mod tests {
    use super::*;
    use burn_tensor::module::conv2d;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn test_conv2d_simple() {
        let batch_size = 2;
        let channels_in = 3;
        let channels_out = 3;
        let kernel_size_1 = 3;
        let kernel_size_2 = 3;
        let padding_1 = 1;
        let padding_2 = 1;
        let stride_1 = 1;
        let stride_2 = 1;
        let height = 6;
        let width = 6;

        let weights = TestTensor::ones([channels_out, channels_in, kernel_size_1, kernel_size_2]);
        let bias = TestTensor::ones([channels_out]);
        let x = TestTensor::ones([batch_size, channels_in, height, width]);
        let y = TestTensor::from_floats([
            [
                [
                    [13., 19., 19., 19., 19., 13.],
                    [19., 28., 28., 28., 28., 19.],
                    [19., 28., 28., 28., 28., 19.],
                    [19., 28., 28., 28., 28., 19.],
                    [19., 28., 28., 28., 28., 19.],
                    [13., 19., 19., 19., 19., 13.],
                ],
                [
                    [13., 19., 19., 19., 19., 13.],
                    [19., 28., 28., 28., 28., 19.],
                    [19., 28., 28., 28., 28., 19.],
                    [19., 28., 28., 28., 28., 19.],
                    [19., 28., 28., 28., 28., 19.],
                    [13., 19., 19., 19., 19., 13.],
                ],
                [
                    [13., 19., 19., 19., 19., 13.],
                    [19., 28., 28., 28., 28., 19.],
                    [19., 28., 28., 28., 28., 19.],
                    [19., 28., 28., 28., 28., 19.],
                    [19., 28., 28., 28., 28., 19.],
                    [13., 19., 19., 19., 19., 13.],
                ],
            ],
            [
                [
                    [13., 19., 19., 19., 19., 13.],
                    [19., 28., 28., 28., 28., 19.],
                    [19., 28., 28., 28., 28., 19.],
                    [19., 28., 28., 28., 28., 19.],
                    [19., 28., 28., 28., 28., 19.],
                    [13., 19., 19., 19., 19., 13.],
                ],
                [
                    [13., 19., 19., 19., 19., 13.],
                    [19., 28., 28., 28., 28., 19.],
                    [19., 28., 28., 28., 28., 19.],
                    [19., 28., 28., 28., 28., 19.],
                    [19., 28., 28., 28., 28., 19.],
                    [13., 19., 19., 19., 19., 13.],
                ],
                [
                    [13., 19., 19., 19., 19., 13.],
                    [19., 28., 28., 28., 28., 19.],
                    [19., 28., 28., 28., 28., 19.],
                    [19., 28., 28., 28., 28., 19.],
                    [19., 28., 28., 28., 28., 19.],
                    [13., 19., 19., 19., 19., 13.],
                ],
            ],
        ]);

        let output = conv2d(
            &x,
            &weights,
            Some(&bias),
            [stride_1, stride_2],
            [padding_1, padding_2],
        );

        y.to_data().assert_approx_eq(&output.into_data(), 3);
    }

    #[test]
    fn test_conv2d_complex() {
        let batch_size = 2;
        let channels_in = 3;
        let channels_out = 4;
        let kernel_size_1 = 3;
        let kernel_size_2 = 2;
        let padding_1 = 1;
        let padding_2 = 2;
        let stride_1 = 2;
        let stride_2 = 3;
        let height = 7;
        let width = 9;

        let weights = TestTensor::ones([channels_out, channels_in, kernel_size_1, kernel_size_2]);
        let bias = TestTensor::ones([channels_out]);
        let x = TestTensor::ones([batch_size, channels_in, height, width]);
        let y = TestTensor::from_floats([
            [
                [
                    [1., 13., 13., 13.],
                    [1., 19., 19., 19.],
                    [1., 19., 19., 19.],
                    [1., 13., 13., 13.],
                ],
                [
                    [1., 13., 13., 13.],
                    [1., 19., 19., 19.],
                    [1., 19., 19., 19.],
                    [1., 13., 13., 13.],
                ],
                [
                    [1., 13., 13., 13.],
                    [1., 19., 19., 19.],
                    [1., 19., 19., 19.],
                    [1., 13., 13., 13.],
                ],
                [
                    [1., 13., 13., 13.],
                    [1., 19., 19., 19.],
                    [1., 19., 19., 19.],
                    [1., 13., 13., 13.],
                ],
            ],
            [
                [
                    [1., 13., 13., 13.],
                    [1., 19., 19., 19.],
                    [1., 19., 19., 19.],
                    [1., 13., 13., 13.],
                ],
                [
                    [1., 13., 13., 13.],
                    [1., 19., 19., 19.],
                    [1., 19., 19., 19.],
                    [1., 13., 13., 13.],
                ],
                [
                    [1., 13., 13., 13.],
                    [1., 19., 19., 19.],
                    [1., 19., 19., 19.],
                    [1., 13., 13., 13.],
                ],
                [
                    [1., 13., 13., 13.],
                    [1., 19., 19., 19.],
                    [1., 19., 19., 19.],
                    [1., 13., 13., 13.],
                ],
            ],
        ]);

        let output = conv2d(
            &x,
            &weights,
            Some(&bias),
            [stride_1, stride_2],
            [padding_1, padding_2],
        );

        y.to_data().assert_approx_eq(&output.into_data(), 3);
    }
}
