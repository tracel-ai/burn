#[burn_tensor_testgen::testgen(module_deform_conv2d)]
mod tests {

    use super::*;
    use burn_tensor::module::deform_conv2d;
    use burn_tensor::ops::{DeformConv2dBackward, DeformConvOptions, ModuleOps};
    use burn_tensor::{Shape, Tensor};

    #[test]
    fn test_deform_conv2d_simple() {
        let test = DeformConv2dTestCase {
            batch_size: 1,
            channels_in: 3,
            channels_out: 5,
            kernel_size_1: 3,
            kernel_size_2: 3,
            padding_1: 0,
            padding_2: 0,
            stride_1: 1,
            stride_2: 1,
            dilation_1: 1,
            dilation_2: 1,
            weight_groups: 1,
            offset_groups: 1,
            height: 4,
            width: 4,
        };

        test.assert_output(Tensor::<TestBackend, 4>::from([[
            [[0.9074, 0.6387], [0.5160, 0.4196]],
            [[2.4259, 1.8008], [1.5449, 1.3112]],
            [[3.9444, 2.9629], [2.5738, 2.2027]],
            [[5.4629, 4.1250], [3.6027, 3.0943]],
            [[6.9814, 5.2871], [4.6316, 3.9859]],
        ]]));
    }

    #[test]
    fn test_deform_conv2d_batched() {
        let test = DeformConv2dTestCase {
            batch_size: 2,
            channels_in: 3,
            channels_out: 5,
            kernel_size_1: 3,
            kernel_size_2: 3,
            padding_1: 0,
            padding_2: 0,
            stride_1: 1,
            stride_2: 1,
            dilation_1: 1,
            dilation_2: 1,
            weight_groups: 1,
            offset_groups: 1,
            height: 4,
            width: 4,
        };

        test.assert_output(Tensor::<TestBackend, 4>::from([
            [
                [[0.2155, 0.1928], [0.1934, 0.1755]],
                [[0.7251, 0.6759], [0.6877, 0.6485]],
                [[1.2347, 1.1590], [1.1821, 1.1215]],
                [[1.7443, 1.6421], [1.6764, 1.5945]],
                [[2.2539, 2.1252], [2.1708, 2.0675]],
            ],
            [
                [[1.6530, 1.1369], [0.9840, 0.7184]],
                [[4.8368, 3.4725], [3.1773, 2.4180]],
                [[8.0206, 5.8080], [5.3705, 4.1176]],
                [[11.2045, 8.1435], [7.5637, 5.8173]],
                [[14.3883, 10.4790], [9.7570, 7.5169]],
            ],
        ]))
    }

    #[test]
    fn test_deform_conv2d_weight_groups() {
        let test = DeformConv2dTestCase {
            batch_size: 1,
            channels_in: 3,
            channels_out: 6,
            kernel_size_1: 3,
            kernel_size_2: 3,
            padding_1: 0,
            padding_2: 0,
            stride_1: 1,
            stride_2: 1,
            dilation_1: 1,
            dilation_2: 1,
            weight_groups: 3,
            offset_groups: 1,
            height: 4,
            width: 4,
        };

        test.assert_output(Tensor::<TestBackend, 4>::from([[
            [[0.1018, 0.0658], [0.0467, 0.0362]],
            [[0.4125, 0.3367], [0.3069, 0.2824]],
            [[1.3076, 1.0242], [0.9025, 0.8000]],
            [[1.8405, 1.4581], [1.2994, 1.1588]],
            [[3.4022, 2.6346], [2.3052, 2.0143]],
            [[4.1574, 3.2315], [2.8389, 2.4857]],
        ]]))
    }

    #[test]
    fn test_deform_conv2d_offset_groups() {
        let test = DeformConv2dTestCase {
            batch_size: 1,
            channels_in: 3,
            channels_out: 6,
            kernel_size_1: 3,
            kernel_size_2: 3,
            padding_1: 0,
            padding_2: 0,
            stride_1: 1,
            stride_2: 1,
            dilation_1: 1,
            dilation_2: 1,
            weight_groups: 1,
            offset_groups: 3,
            height: 4,
            width: 4,
        };

        test.assert_output(Tensor::<TestBackend, 4>::from([[
            [[1.0794, 0.7676], [0.7209, 0.5337]],
            [[2.7059, 2.0216], [1.9740, 1.5419]],
            [[4.3325, 3.2755], [3.2271, 2.5501]],
            [[5.9590, 4.5295], [4.4802, 3.5582]],
            [[7.5855, 5.7835], [5.7333, 4.5664]],
            [[9.2120, 7.0375], [6.9864, 5.5746]],
        ]]))
    }

    #[test]
    fn test_deform_conv2d_different_kernel_size() {
        let test = DeformConv2dTestCase {
            batch_size: 1,
            channels_in: 2,
            channels_out: 3,
            kernel_size_1: 3,
            kernel_size_2: 4,
            padding_1: 0,
            padding_2: 0,
            stride_1: 1,
            stride_2: 1,
            dilation_1: 1,
            dilation_2: 1,
            weight_groups: 1,
            offset_groups: 1,
            height: 4,
            width: 4,
        };

        test.assert_output(Tensor::<TestBackend, 4>::from([[
            [[1.0669], [0.6329]],
            [[2.9741], [2.0383]],
            [[4.8812], [3.4437]],
        ]]))
    }

    #[test]
    fn test_deform_conv2d_different_padding_size() {
        let test = DeformConv2dTestCase {
            batch_size: 1,
            channels_in: 2,
            channels_out: 3,
            kernel_size_1: 3,
            kernel_size_2: 3,
            padding_1: 2,
            padding_2: 3,
            stride_1: 1,
            stride_2: 1,
            dilation_1: 1,
            dilation_2: 1,
            weight_groups: 1,
            offset_groups: 1,
            height: 4,
            width: 4,
        };

        test.assert_output(Tensor::<TestBackend, 4>::from([[
            [
                [
                    0.1998, 0.3762, 0.5285, 0.6053, 0.3844, 0.1987, 0.0481, 0.0000,
                ],
                [
                    0.2879, 0.5517, 0.7776, 0.8905, 0.5805, 0.3043, 0.0796, 0.0000,
                ],
                [
                    0.3729, 0.7214, 1.0137, 1.1520, 0.7564, 0.3931, 0.1016, 0.0000,
                ],
                [
                    0.1321, 0.3249, 0.4954, 0.5846, 0.4531, 0.2501, 0.0757, 0.0000,
                ],
                [
                    0.0593, 0.1607, 0.2448, 0.2971, 0.2395, 0.1327, 0.0471, 0.0000,
                ],
                [
                    0.0143, 0.0513, 0.0783, 0.0942, 0.0813, 0.0420, 0.0145, 0.0000,
                ],
            ],
            [
                [
                    0.7667, 1.1648, 1.5219, 1.7111, 1.2305, 0.8076, 0.4504, 0.3333,
                ],
                [
                    0.9812, 1.6010, 2.1525, 2.4409, 1.7455, 1.0918, 0.5367, 0.3333,
                ],
                [
                    1.1964, 2.0448, 2.7853, 3.1522, 2.2426, 1.3513, 0.6049, 0.3333,
                ],
                [
                    0.6695, 1.1781, 1.6441, 1.9022, 1.5732, 1.0339, 0.5536, 0.3333,
                ],
                [
                    0.4950, 0.7861, 1.0398, 1.2047, 1.0523, 0.7439, 0.4834, 0.3333,
                ],
                [
                    0.3788, 0.4982, 0.5929, 0.6542, 0.6155, 0.4882, 0.3909, 0.3333,
                ],
            ],
            [
                [
                    1.3335, 1.9534, 2.5154, 2.8170, 2.0766, 1.4165, 0.8527, 0.6667,
                ],
                [
                    1.6744, 2.6503, 3.5275, 3.9914, 2.9106, 1.8794, 0.9939, 0.6667,
                ],
                [
                    2.0198, 3.3683, 4.5570, 5.1525, 3.7288, 2.3095, 1.1082, 0.6667,
                ],
                [
                    1.2068, 2.0314, 2.7928, 3.2198, 2.6932, 1.8178, 1.0315, 0.6667,
                ],
                [
                    0.9308, 1.4116, 1.8348, 2.1124, 1.8652, 1.3551, 0.9196, 0.6667,
                ],
                [
                    0.7432, 0.9451, 1.1074, 1.2143, 1.1497, 0.9345, 0.7673, 0.6667,
                ],
            ],
        ]]))
    }

    #[test]
    fn test_deform_conv2d_different_stride() {
        let test = DeformConv2dTestCase {
            batch_size: 1,
            channels_in: 2,
            channels_out: 3,
            kernel_size_1: 3,
            kernel_size_2: 3,
            padding_1: 0,
            padding_2: 0,
            stride_1: 1,
            stride_2: 2,
            dilation_1: 1,
            dilation_2: 1,
            weight_groups: 1,
            offset_groups: 1,
            height: 4,
            width: 4,
        };

        test.assert_output(Tensor::<TestBackend, 4>::from([[
            [[1.0647], [0.5783]],
            [[2.9289], [1.8829]],
            [[4.7931], [3.1875]],
        ]]))
    }

    #[test]
    fn test_deform_conv2d_different_dilation() {
        let test = DeformConv2dTestCase {
            batch_size: 1,
            channels_in: 2,
            channels_out: 3,
            kernel_size_1: 3,
            kernel_size_2: 3,
            padding_1: 0,
            padding_2: 0,
            stride_1: 1,
            stride_2: 1,
            dilation_1: 1,
            dilation_2: 2,
            weight_groups: 1,
            offset_groups: 1,
            height: 5,
            width: 5,
        };

        test.assert_output(Tensor::<TestBackend, 4>::from([[
            [[0.6162], [0.7611], [0.4666]],
            [[1.8578], [2.2684], [1.6208]],
            [[3.0994], [3.7757], [2.7749]],
        ]]))
    }

    #[test]
    fn test_deform_conv2d_different_width() {
        let test = DeformConv2dTestCase {
            batch_size: 1,
            channels_in: 2,
            channels_out: 3,
            kernel_size_1: 3,
            kernel_size_2: 3,
            padding_1: 0,
            padding_2: 0,
            stride_1: 1,
            stride_2: 1,
            dilation_1: 1,
            dilation_2: 1,
            weight_groups: 1,
            offset_groups: 1,
            height: 6,
            width: 4,
        };

        test.assert_output(Tensor::<TestBackend, 4>::from([[
            [
                [0.8909, 0.6016],
                [1.0697, 0.7186],
                [1.2618, 0.8433],
                [0.6424, 0.5032],
            ],
            [
                [2.4670, 1.8168],
                [2.9529, 2.1497],
                [3.4805, 2.5090],
                [2.0925, 1.7411],
            ],
            [
                [4.0432, 3.0321],
                [4.8362, 3.5809],
                [5.6992, 4.1746],
                [3.5425, 2.9790],
            ],
        ]]))
    }

    struct DeformConv2dTestCase {
        batch_size: usize,
        channels_in: usize,
        channels_out: usize,
        kernel_size_1: usize,
        kernel_size_2: usize,
        padding_1: usize,
        padding_2: usize,
        stride_1: usize,
        stride_2: usize,
        dilation_1: usize,
        dilation_2: usize,
        weight_groups: usize,
        offset_groups: usize,
        height: usize,
        width: usize,
    }

    impl DeformConv2dTestCase {
        fn assert_output(self, y: Tensor<TestBackend, 4>) {
            let out_height =
                (self.height + 2 * self.padding_1 - self.dilation_1 * (self.kernel_size_1 - 1) - 1)
                    / self.stride_1
                    + 1;
            let out_width =
                (self.width + 2 * self.padding_2 - self.dilation_2 * (self.kernel_size_2 - 1) - 1)
                    / self.stride_2
                    + 1;

            let shape_x = Shape::new([self.batch_size, self.channels_in, self.height, self.width]);
            let shape_weight = Shape::new([
                self.channels_out,
                self.channels_in / self.weight_groups,
                self.kernel_size_1,
                self.kernel_size_2,
            ]);
            let shape_offset = Shape::new([
                self.batch_size,
                self.kernel_size_1 * self.kernel_size_2 * self.offset_groups * 2,
                out_height,
                out_width,
            ]);
            let shape_mask = Shape::new([
                self.batch_size,
                self.kernel_size_1 * self.kernel_size_2 * self.offset_groups,
                out_height,
                out_width,
            ]);
            let device = Default::default();
            let weight = Tensor::<TestBackend, 4>::from(
                TestTensorInt::arange(0..shape_weight.num_elements() as i64, &device)
                    .reshape::<4, _>(shape_weight.clone())
                    .into_data(),
            )
            .div_scalar(shape_weight.num_elements() as f32);
            let bias = Tensor::<TestBackend, 1>::from(
                TestTensorInt::arange(0..self.channels_out as i64, &device).into_data(),
            )
            .div_scalar(self.channels_out as f32);
            let x = Tensor::<TestBackend, 4>::from(
                TestTensorInt::arange(0..shape_x.num_elements() as i64, &device)
                    .reshape::<4, _>(shape_x.clone())
                    .into_data(),
            )
            .div_scalar(shape_x.num_elements() as f32);
            let offset = Tensor::<TestBackend, 4>::from(
                TestTensorInt::arange(0..shape_offset.num_elements() as i64, &device)
                    .reshape::<4, _>(shape_offset.clone())
                    .into_data(),
            )
            .div_scalar(shape_offset.num_elements() as f32);
            let mask = Tensor::<TestBackend, 4>::from(
                TestTensorInt::arange(0..shape_mask.num_elements() as i64, &device)
                    .reshape::<4, _>(shape_mask.clone())
                    .into_data(),
            )
            .div_scalar(shape_mask.num_elements() as f32);

            let output = deform_conv2d(
                x,
                offset,
                weight,
                Some(mask),
                Some(bias),
                DeformConvOptions::new(
                    [self.stride_1, self.stride_2],
                    [self.padding_1, self.padding_2],
                    [self.dilation_1, self.dilation_2],
                    self.weight_groups,
                    self.offset_groups,
                ),
            );

            y.to_data().assert_approx_eq(&output.into_data(), 3);
        }
    }
}
