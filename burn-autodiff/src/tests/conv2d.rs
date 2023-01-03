#[burn_tensor_testgen::testgen(ad_conv2d)]
mod tests {
    use super::*;
    use burn_tensor::{module::conv2d, Data};

    #[test]
    fn test_conv2d_backward_simple_1() {
        // Data
        let x = TestADTensor::from_floats([[[
            [-0.0497, -0.1648, -0.1648, -0.1370],
            [0.6010, 0.6706, 0.6706, 0.3665],
            [0.6761, 0.7699, 0.7699, 0.1930],
        ]]]);
        let y = TestADTensor::from_floats([[[
            [12.3413, 12.5760, 12.5123, 12.3291],
            [12.1590, 12.5360, 12.5361, 12.4187],
            [11.8841, 11.8790, 11.9428, 12.0358],
        ]]]);
        let weight = TestADTensor::from_floats([[[
            [-0.2728, 0.1977, -0.0243],
            [0.2450, -0.2196, -0.0908],
            [0.3319, 0.3188, 0.1846],
        ]]]);
        let bias = TestADTensor::from_floats([12.]);
        let x_grad = TestADTensor::from_floats([[[
            [-0.0497, -0.1648, -0.1648, -0.1370],
            [0.6010, 0.6704, 0.6704, 0.3664],
            [0.6761, 0.7699, 0.7699, 0.1930],
        ]]]);
        let weight_grad = TestADTensor::from_floats([[[
            [1.5629, 1.7924, 1.2411],
            [3.7788, 4.2013, 2.9739],
            [4.1581, 4.7176, 3.4405],
        ]]]);
        let bias_grad = TestADTensor::from_floats([12.]);

        // Convolution
        let output = conv2d(&x, &weight, Some(&bias), [1, 1], [1, 1]);
        let grads = output.backward();

        // Assert
        let x_grad_actual = x.grad(&grads).unwrap();
        let weight_grad_actual = weight.grad(&grads).unwrap();
        let bias_grad_actual = bias.grad(&grads).unwrap();

        y.to_data().assert_approx_eq(&output.to_data(), 3);
        weight_grad
            .to_data()
            .assert_approx_eq(&weight_grad_actual.to_data(), 3);
        x_grad
            .to_data()
            .assert_approx_eq(&x_grad_actual.to_data(), 3);
        bias_grad
            .to_data()
            .assert_approx_eq(&bias_grad_actual.to_data(), 3);
    }

    #[test]
    fn test_conv2d_backward_simple_2() {
        // Data
        let weight = TestADTensor::from_floats([[
            [
                [0.1658, 0.1734, -0.0314],
                [0.1862, -0.1822, -0.1361],
                [-0.0125, -0.0802, 0.1406],
            ],
            [
                [-0.0225, -0.1251, -0.1164],
                [0.1501, -0.0076, 0.0184],
                [0.1083, -0.1609, 0.1634],
            ],
            [
                [0.1552, -0.1230, 0.1840],
                [0.0776, -0.1885, -0.0355],
                [0.1811, -0.1006, -0.0992],
            ],
        ]]);
        let bias = TestADTensor::from_floats([0.1052]);
        let x = TestADTensor::from_floats([[
            [
                [0.4641, 0.6162, 0.7312, 0.7367],
                [0.6886, 0.2303, 0.3891, 0.0685],
                [0.6244, 0.9966, 0.5988, 0.5025],
            ],
            [
                [0.7877, 0.2563, 0.1434, 0.4554],
                [0.9788, 0.4391, 0.1347, 0.1773],
                [0.0923, 0.8667, 0.2041, 0.4296],
            ],
            [
                [0.3147, 0.1842, 0.3693, 0.5453],
                [0.9141, 0.2347, 0.2589, 0.0997],
                [0.3174, 0.1519, 0.5819, 0.5737],
            ],
        ]]);
        let y = TestADTensor::from_floats([[[
            [-0.3542, 0.2766, -0.0056, 0.0636],
            [-0.1339, 0.3754, 0.3679, 0.3172],
            [-0.3248, 0.1546, 0.2105, 0.1686],
        ]]]);

        let x_grad = TestADTensor::from_floats([[
            [
                [0.3432, 0.1757, 0.1757, -0.1763],
                [0.2506, 0.2236, 0.2236, -0.1159],
                [-0.0887, -0.0842, -0.0842, -0.2579],
            ],
            [
                [-0.0051, -0.1031, -0.1031, -0.2307],
                [-0.0577, 0.0077, 0.0077, -0.2282],
                [0.0899, 0.2717, 0.2717, 0.0133],
            ],
            [
                [-0.0787, 0.0698, 0.0698, -0.1630],
                [0.0018, 0.0511, 0.0511, -0.3628],
                [-0.0304, -0.1651, -0.1651, -0.4238],
            ],
        ]]);
        let weight_grad = TestADTensor::from_floats([[
            [
                [3.1195, 3.9247, 2.7720],
                [5.3393, 6.6470, 4.8699],
                [3.5278, 4.0988, 2.7858],
            ],
            [
                [2.7400, 3.3727, 1.6062],
                [3.9031, 4.9654, 3.1066],
                [2.7157, 3.3226, 2.2516],
            ],
            [
                [2.2759, 2.9209, 1.6921],
                [3.3271, 4.5458, 2.9996],
                [2.4589, 3.1323, 1.9008],
            ],
        ]]);
        let bias_grad = TestADTensor::from_floats([12.]);

        // Convolution
        let output = conv2d(&x, &weight, Some(&bias), [1, 1], [1, 1]);
        let grads = output.backward();

        // Assert
        let x_grad_actual = x.grad(&grads).unwrap();
        let weight_grad_actual = weight.grad(&grads).unwrap();
        let bias_grad_actual = bias.grad(&grads).unwrap();

        y.to_data().assert_approx_eq(&output.to_data(), 3);
        weight_grad
            .to_data()
            .assert_approx_eq(&weight_grad_actual.to_data(), 3);
        x_grad
            .to_data()
            .assert_approx_eq(&x_grad_actual.to_data(), 3);
        bias_grad
            .to_data()
            .assert_approx_eq(&bias_grad_actual.to_data(), 3);
    }
}
