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

        test.assert_output(TestTensor::from([[
            [[3.5122, 3.7539], [2.6651, 2.1670]],
            [[3.0452, 2.5101], [2.7505, 1.8762]],
            [[2.5257, 2.7189], [2.1275, 1.6800]],
            [[2.6337, 2.9503], [2.5916, 1.9741]],
            [[2.6041, 2.3439], [2.6403, 1.8981]],
        ]]));
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
        fn assert_output(self, y: TestTensor<4>) {
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
                self.kernel_size_1 * self.kernel_size_2 * self.offset_groups * 2,
                out_height,
                out_width,
            ]);
            let device = Default::default();
            let weight = TestTensor::from(
                TestTensorInt::arange(0..shape_weight.num_elements() as i64, &device)
                    .reshape(shape_weight)
                    .into_data(),
            );
            let bias = TestTensor::from(
                TestTensorInt::arange(0..self.channels_out as i64, &device).into_data(),
            );
            let x = TestTensor::from(
                TestTensorInt::arange(0..shape_x.num_elements() as i64, &device)
                    .reshape(shape_x)
                    .into_data(),
            );
            let offset = TestTensor::from(
                TestTensorInt::arange(0..shape_offset.num_elements() as i64, &device)
                    .reshape(shape_offset)
                    .into_data(),
            );
            let mask = TestTensor::from(
                TestTensorInt::arange(0..shape_mask.num_elements() as i64, &device)
                    .reshape(shape_mask)
                    .into_data(),
            );
            println!("test input: {:?}", x);
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
