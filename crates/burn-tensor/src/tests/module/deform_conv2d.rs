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
            [[122.4954, 86.2217], [69.6655, 56.6430]],
            [[301.4954, 217.1068], [182.5671, 151.0068]],
            [[480.4954, 347.9919], [295.4688, 245.3705]],
            [[659.4954, 478.8770], [408.3705, 339.7342]],
            [[838.4954, 609.7621], [521.2722, 434.0979]],
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
        fn assert_output(self, y: Tensor<TestBackend, 4>) {
            let out_height =
                (self.height + 2 * self.padding_1 - self.dilation_1 * (self.kernel_size_1 - 1) - 1)
                    / self.stride_1
                    + 1;
            let out_width =
                (self.width + 2 * self.padding_2 - self.dilation_2 * (self.kernel_size_2 - 1) - 1)
                    / self.stride_2
                    + 1;

            println!("Out dims: ({out_width}x{out_height})");

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
            );
            let bias = Tensor::<TestBackend, 1>::from(
                TestTensorInt::arange(0..self.channels_out as i64, &device).into_data(),
            );
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
