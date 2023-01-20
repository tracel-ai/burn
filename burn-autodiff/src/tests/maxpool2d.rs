#[burn_tensor_testgen::testgen(ad_max_pool2d)]
mod tests {
    use super::*;
    use burn_tensor::{module::max_pool2d, Data};

    #[test]
    fn test_max_pool2d_simple() {
        let batch_size = 1;
        let channels_in = 1;
        let kernel_size_1 = 2;
        let kernel_size_2 = 2;
        let padding_1 = 1;
        let padding_2 = 1;
        let stride_1 = 1;
        let stride_2 = 1;

        let x = TestADTensor::from_floats([[[
            [0.2479, 0.6386, 0.3166, 0.5742],
            [0.7065, 0.1940, 0.6305, 0.8959],
            [0.5416, 0.8602, 0.8129, 0.1662],
            [0.3358, 0.3059, 0.8293, 0.0990],
        ]]]);
        let x_grad_expected = TestADTensor::from_floats([[[
            [1., 3., 0., 2.],
            [3., 0., 0., 4.],
            [1., 4., 0., 1.],
            [2., 0., 3., 1.],
        ]]]);

        let output = max_pool2d(
            &x,
            [kernel_size_1, kernel_size_2],
            [stride_1, stride_2],
            [padding_1, padding_2],
        );
        let grads = output.backward();

        // Asserts
        let x_grad_actual = x.grad(&grads).unwrap();
        x_grad_expected
            .to_data()
            .assert_approx_eq(&x_grad_actual.to_data(), 3);
    }

    #[test]
    fn test_max_pool2d_complex() {
        let batch_size = 1;
        let channels_in = 1;
        let kernel_size_1 = 4;
        let kernel_size_2 = 2;
        let padding_1 = 2;
        let padding_2 = 1;
        let stride_1 = 1;
        let stride_2 = 2;

        let x = TestADTensor::from_floats([[[
            [0.5388, 0.0676, 0.7122, 0.8316, 0.0653],
            [0.9154, 0.1536, 0.9089, 0.8016, 0.7518],
            [0.2073, 0.0501, 0.8811, 0.5604, 0.5075],
            [0.4384, 0.9963, 0.9698, 0.4988, 0.2609],
            [0.3391, 0.2230, 0.4610, 0.5365, 0.6880],
        ]]]);
        let x_grad_expected = TestADTensor::from_floats([[[
            [0., 0., 0., 3., 0.],
            [4., 0., 2., 1., 0.],
            [0., 0., 0., 0., 0.],
            [2., 4., 0., 0., 0.],
            [0., 0., 0., 0., 2.],
        ]]]);

        let output = max_pool2d(
            &x,
            [kernel_size_1, kernel_size_2],
            [stride_1, stride_2],
            [padding_1, padding_2],
        );
        let grads = output.backward();

        // Asserts
        let x_grad_actual = x.grad(&grads).unwrap();
        x_grad_expected
            .to_data()
            .assert_approx_eq(&x_grad_actual.to_data(), 3);
    }
}
