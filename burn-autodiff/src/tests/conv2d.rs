#[burn_tensor_testgen::testgen(ad_conv2d)]
mod tests {
    use super::*;
    use burn_tensor::{module::conv2d, Data};

    #[test]
    fn test_conv2d_complex() {
        let test = Conv2dTestCase {
            batch_size: 2,
            channels_in: 3,
            channels_out: 3,
            kernel_size_1: 3,
            kernel_size_2: 3,
            padding_1: 1,
            padding_2: 1,
            stride_1: 1,
            stride_2: 1,
            height: 6,
            width: 6,
        };
        let grads = Grads {
            x: TestTensor::from_floats([
                [
                    [
                        [12., 18., 18., 18., 18., 12.],
                        [18., 27., 27., 27., 27., 18.],
                        [18., 27., 27., 27., 27., 18.],
                        [18., 27., 27., 27., 27., 18.],
                        [18., 27., 27., 27., 27., 18.],
                        [12., 18., 18., 18., 18., 12.],
                    ],
                    [
                        [12., 18., 18., 18., 18., 12.],
                        [18., 27., 27., 27., 27., 18.],
                        [18., 27., 27., 27., 27., 18.],
                        [18., 27., 27., 27., 27., 18.],
                        [18., 27., 27., 27., 27., 18.],
                        [12., 18., 18., 18., 18., 12.],
                    ],
                    [
                        [12., 18., 18., 18., 18., 12.],
                        [18., 27., 27., 27., 27., 18.],
                        [18., 27., 27., 27., 27., 18.],
                        [18., 27., 27., 27., 27., 18.],
                        [18., 27., 27., 27., 27., 18.],
                        [12., 18., 18., 18., 18., 12.],
                    ],
                ],
                [
                    [
                        [12., 18., 18., 18., 18., 12.],
                        [18., 27., 27., 27., 27., 18.],
                        [18., 27., 27., 27., 27., 18.],
                        [18., 27., 27., 27., 27., 18.],
                        [18., 27., 27., 27., 27., 18.],
                        [12., 18., 18., 18., 18., 12.],
                    ],
                    [
                        [12., 18., 18., 18., 18., 12.],
                        [18., 27., 27., 27., 27., 18.],
                        [18., 27., 27., 27., 27., 18.],
                        [18., 27., 27., 27., 27., 18.],
                        [18., 27., 27., 27., 27., 18.],
                        [12., 18., 18., 18., 18., 12.],
                    ],
                    [
                        [12., 18., 18., 18., 18., 12.],
                        [18., 27., 27., 27., 27., 18.],
                        [18., 27., 27., 27., 27., 18.],
                        [18., 27., 27., 27., 27., 18.],
                        [18., 27., 27., 27., 27., 18.],
                        [12., 18., 18., 18., 18., 12.],
                    ],
                ],
            ]),
            weight: TestTensor::from_floats([
                [
                    [[50., 60., 50.], [60., 72., 60.], [50., 60., 50.]],
                    [[50., 60., 50.], [60., 72., 60.], [50., 60., 50.]],
                    [[50., 60., 50.], [60., 72., 60.], [50., 60., 50.]],
                ],
                [
                    [[50., 60., 50.], [60., 72., 60.], [50., 60., 50.]],
                    [[50., 60., 50.], [60., 72., 60.], [50., 60., 50.]],
                    [[50., 60., 50.], [60., 72., 60.], [50., 60., 50.]],
                ],
                [
                    [[50., 60., 50.], [60., 72., 60.], [50., 60., 50.]],
                    [[50., 60., 50.], [60., 72., 60.], [50., 60., 50.]],
                    [[50., 60., 50.], [60., 72., 60.], [50., 60., 50.]],
                ],
            ]),
            bias: TestTensor::from_floats([72., 72., 72.]),
        };
        test.assert_grads(grads);
    }

    #[test]
    fn test_conv2d_different_stride() {
        let test = Conv2dTestCase {
            batch_size: 2,
            channels_in: 2,
            channels_out: 2,
            kernel_size_1: 3,
            kernel_size_2: 3,
            padding_1: 1,
            padding_2: 1,
            stride_1: 1,
            stride_2: 2,
            height: 8,
            width: 8,
        };
        let grads = Grads {
            x: TestTensor::from_floats([
                [
                    [
                        [4., 8., 4., 8., 4., 8., 4., 4.],
                        [6., 12., 6., 12., 6., 12., 6., 6.],
                        [6., 12., 6., 12., 6., 12., 6., 6.],
                        [6., 12., 6., 12., 6., 12., 6., 6.],
                        [6., 12., 6., 12., 6., 12., 6., 6.],
                        [6., 12., 6., 12., 6., 12., 6., 6.],
                        [6., 12., 6., 12., 6., 12., 6., 6.],
                        [4., 8., 4., 8., 4., 8., 4., 4.],
                    ],
                    [
                        [4., 8., 4., 8., 4., 8., 4., 4.],
                        [6., 12., 6., 12., 6., 12., 6., 6.],
                        [6., 12., 6., 12., 6., 12., 6., 6.],
                        [6., 12., 6., 12., 6., 12., 6., 6.],
                        [6., 12., 6., 12., 6., 12., 6., 6.],
                        [6., 12., 6., 12., 6., 12., 6., 6.],
                        [6., 12., 6., 12., 6., 12., 6., 6.],
                        [4., 8., 4., 8., 4., 8., 4., 4.],
                    ],
                ],
                [
                    [
                        [4., 8., 4., 8., 4., 8., 4., 4.],
                        [6., 12., 6., 12., 6., 12., 6., 6.],
                        [6., 12., 6., 12., 6., 12., 6., 6.],
                        [6., 12., 6., 12., 6., 12., 6., 6.],
                        [6., 12., 6., 12., 6., 12., 6., 6.],
                        [6., 12., 6., 12., 6., 12., 6., 6.],
                        [6., 12., 6., 12., 6., 12., 6., 6.],
                        [4., 8., 4., 8., 4., 8., 4., 4.],
                    ],
                    [
                        [4., 8., 4., 8., 4., 8., 4., 4.],
                        [6., 12., 6., 12., 6., 12., 6., 6.],
                        [6., 12., 6., 12., 6., 12., 6., 6.],
                        [6., 12., 6., 12., 6., 12., 6., 6.],
                        [6., 12., 6., 12., 6., 12., 6., 6.],
                        [6., 12., 6., 12., 6., 12., 6., 6.],
                        [6., 12., 6., 12., 6., 12., 6., 6.],
                        [4., 8., 4., 8., 4., 8., 4., 4.],
                    ],
                ],
            ]),
            weight: TestTensor::from_floats([
                [
                    [[42., 56., 56.], [48., 64., 64.], [42., 56., 56.]],
                    [[42., 56., 56.], [48., 64., 64.], [42., 56., 56.]],
                ],
                [
                    [[42., 56., 56.], [48., 64., 64.], [42., 56., 56.]],
                    [[42., 56., 56.], [48., 64., 64.], [42., 56., 56.]],
                ],
            ]),
            bias: TestTensor::from_floats([64., 64.]),
        };
        test.assert_grads(grads);
    }

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

    #[test]
    fn test_conv2d_backward_simple_3() {
        // Data
        let weight = TestADTensor::from_floats([
            [
                [
                    [0.0106, 0.1603, -0.1523],
                    [0.0230, 0.1315, -0.1599],
                    [-0.1779, -0.1336, 0.1794],
                ],
                [
                    [0.1034, -0.1766, 0.1278],
                    [0.1300, 0.1922, 0.0063],
                    [0.1105, 0.0156, -0.0847],
                ],
                [
                    [0.0827, -0.0755, 0.0970],
                    [-0.1759, 0.0081, 0.1799],
                    [0.1251, 0.0365, -0.1434],
                ],
            ],
            [
                [
                    [-0.0158, 0.0380, -0.0514],
                    [-0.0207, 0.0980, 0.1105],
                    [-0.1767, 0.0084, 0.0584],
                ],
                [
                    [-0.0223, -0.0560, 0.0837],
                    [0.1445, 0.1762, -0.0569],
                    [-0.1656, 0.1512, 0.0723],
                ],
                [
                    [0.0521, 0.1749, -0.0533],
                    [-0.0630, -0.1043, -0.0778],
                    [-0.1171, 0.0234, 0.0525],
                ],
            ],
        ]);
        let bias = TestADTensor::from_floats([0.0089, -0.1223]);
        let x = TestADTensor::from_floats([[
            [
                [0.3071, 0.8465, 0.4366, 0.9659],
                [0.4102, 0.5293, 0.5801, 0.8459],
                [0.5416, 0.3806, 0.3913, 0.3121],
            ],
            [
                [0.2261, 0.6681, 0.3344, 0.8660],
                [0.1861, 0.9620, 0.2702, 0.1643],
                [0.9096, 0.1231, 0.9576, 0.6170],
            ],
            [
                [0.3941, 0.6263, 0.9472, 0.4891],
                [0.4281, 0.9239, 0.5189, 0.5886],
                [0.9535, 0.6737, 0.5185, 0.4279],
            ],
        ]]);
        let y = TestADTensor::from_floats([[
            [
                [-0.0778, 0.3094, 0.2002, 0.0958],
                [0.1243, 0.3431, 0.1221, 0.2808],
                [0.4563, -0.0114, 0.3227, 0.3534],
            ],
            [
                [0.1039, 0.0688, -0.2714, -0.1087],
                [0.1158, -0.1286, 0.2799, -0.1513],
                [0.0583, -0.0204, 0.0191, 0.2077],
            ],
        ]]);

        let x_grad = TestADTensor::from_floats([[
            [
                [0.4249, 0.1718, 0.1718, 0.1747],
                [-0.0549, -0.0702, -0.0702, 0.2873],
                [-0.2480, -0.0596, -0.0596, 0.2927],
            ],
            [
                [0.4914, 0.6523, 0.6523, 0.2967],
                [0.6031, 0.7516, 0.7516, 0.4511],
                [0.7546, 0.6916, 0.6916, 0.4722],
            ],
            [
                [-0.1009, 0.0449, 0.0449, 0.1490],
                [-0.0330, 0.0219, 0.0219, 0.1180],
                [-0.2672, -0.2560, -0.2560, -0.0251],
            ],
        ]]);
        let weight_grad = TestADTensor::from_floats([
            [
                [
                    [3.1098, 4.9216, 4.2043],
                    [4.4233, 6.5472, 5.2883],
                    [2.8331, 3.9911, 3.0393],
                ],
                [
                    [2.6469, 3.6772, 3.2650],
                    [4.6372, 6.2845, 4.9627],
                    [3.4086, 4.1899, 3.0942],
                ],
                [
                    [3.8385, 4.9162, 4.0940],
                    [5.9842, 7.4898, 5.7141],
                    [4.0166, 5.0331, 3.6515],
                ],
            ],
            [
                [
                    [3.1098, 4.9216, 4.2043],
                    [4.4233, 6.5472, 5.2883],
                    [2.8331, 3.9911, 3.0393],
                ],
                [
                    [2.6469, 3.6772, 3.2650],
                    [4.6372, 6.2845, 4.9627],
                    [3.4086, 4.1899, 3.0942],
                ],
                [
                    [3.8385, 4.9162, 4.0940],
                    [5.9842, 7.4898, 5.7141],
                    [4.0166, 5.0331, 3.6515],
                ],
            ],
        ]);
        let bias_grad = TestADTensor::from_floats([12., 12.]);

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

    struct Conv2dTestCase {
        batch_size: usize,
        channels_in: usize,
        channels_out: usize,
        kernel_size_1: usize,
        kernel_size_2: usize,
        padding_1: usize,
        padding_2: usize,
        stride_1: usize,
        stride_2: usize,
        height: usize,
        width: usize,
    }

    struct Grads {
        x: TestTensor<4>,
        weight: TestTensor<4>,
        bias: TestTensor<1>,
    }

    impl Conv2dTestCase {
        fn assert_grads(self, expected_grads: Grads) {
            let weight = TestADTensor::ones([
                self.channels_out,
                self.channels_in,
                self.kernel_size_1,
                self.kernel_size_2,
            ]);
            let bias = TestADTensor::ones([self.channels_out]);
            let x =
                TestADTensor::ones([self.batch_size, self.channels_in, self.height, self.width]);
            let output = conv2d(
                &x,
                &weight,
                Some(&bias),
                [self.stride_1, self.stride_2],
                [self.padding_1, self.padding_2],
            );
            let grads = output.backward();

            // Assert
            let x_grad_actual = x.grad(&grads).unwrap();
            let weight_grad_actual = weight.grad(&grads).unwrap();
            let bias_grad_actual = bias.grad(&grads).unwrap();

            println!("x");
            expected_grads
                .x
                .to_data()
                .assert_approx_eq(&x_grad_actual.to_data(), 3);
            println!("weight");
            expected_grads
                .weight
                .to_data()
                .assert_approx_eq(&weight_grad_actual.to_data(), 3);
            println!("bias");
            expected_grads
                .bias
                .to_data()
                .assert_approx_eq(&bias_grad_actual.to_data(), 3);
        }
    }
}
