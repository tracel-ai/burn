#[burn_tensor_testgen::testgen(module_deform_conv2d)]
mod tests {
    use super::*;
    use burn_tensor::module::deform_conv2d;
    use burn_tensor::ops::DeformConvOptions;
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
    #[ignore = "Need to figure out why it's wrong"]
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
                    .reshape(shape_weight.clone())
                    .into_data(),
            )
            .div_scalar(shape_weight.num_elements() as f32);
            let bias = Tensor::<TestBackend, 1>::from(
                TestTensorInt::arange(0..self.channels_out as i64, &device).into_data(),
            )
            .div_scalar(self.channels_out as f32);
            let x = Tensor::<TestBackend, 4>::from(
                TestTensorInt::arange(0..shape_x.num_elements() as i64, &device)
                    .reshape(shape_x.clone())
                    .into_data(),
            )
            .div_scalar(shape_x.num_elements() as f32);
            let offset = Tensor::<TestBackend, 4>::from(
                TestTensorInt::arange(0..shape_offset.num_elements() as i64, &device)
                    .reshape(shape_offset.clone())
                    .into_data(),
            )
            .div_scalar(shape_offset.num_elements() as f32);
            let mask = Tensor::<TestBackend, 4>::from(
                TestTensorInt::arange(0..shape_mask.num_elements() as i64, &device)
                    .reshape(shape_mask.clone())
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
