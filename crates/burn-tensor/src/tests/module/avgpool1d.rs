#[burn_tensor_testgen::testgen(module_avg_pool1d)]
mod tests {
    use super::*;
    use burn_tensor::module::avg_pool1d;
    use burn_tensor::{Shape, Tensor};
    use burn_tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn test_avg_pool1d_simple() {
        let test = AvgPool1dTestCase {
            batch_size: 1,
            channels: 1,
            kernel_size: 3,
            padding: 0,
            stride: 1,
            length: 6,
            count_include_pad: true,
        };

        test.assert_output(TestTensor::from([[[1., 2., 3., 4.]]]));
    }

    #[test]
    fn test_avg_pool1d_complex() {
        let test = AvgPool1dTestCase {
            batch_size: 1,
            channels: 2,
            kernel_size: 3,
            padding: 1,
            stride: 2,
            length: 6,
            count_include_pad: true,
        };

        test.assert_output(TestTensor::from([[
            [0.33333, 2.0000, 4.0000],
            [4.33333, 8.0000, 10.0000],
        ]]));
    }

    #[test]
    fn test_avg_pool1d_complex_dont_count_pad() {
        let test = AvgPool1dTestCase {
            batch_size: 1,
            channels: 2,
            kernel_size: 3,
            padding: 1,
            stride: 2,
            length: 6,
            count_include_pad: false,
        };

        test.assert_output(TestTensor::from([[
            [0.5000, 2.0000, 4.0000],
            [6.5000, 8.0000, 10.0000],
        ]]));
    }

    struct AvgPool1dTestCase {
        batch_size: usize,
        channels: usize,
        kernel_size: usize,
        padding: usize,
        stride: usize,
        length: usize,
        count_include_pad: bool,
    }

    impl AvgPool1dTestCase {
        fn assert_output(self, y: TestTensor<3>) {
            let shape_x = Shape::new([self.batch_size, self.channels, self.length]);
            let x = TestTensor::from(
                TestTensorInt::arange(0..shape_x.num_elements() as i64, &y.device())
                    .reshape::<3, _>(shape_x)
                    .into_data(),
            );
            let output = avg_pool1d(
                x,
                self.kernel_size,
                self.stride,
                self.padding,
                self.count_include_pad,
                false,
            );

            y.to_data().assert_approx_eq::<FT>(
                &output.into_data(),
                Tolerance::default().set_half_precision_relative(1e-3),
            );
        }
    }

    #[test]
    fn test_avg_pool1d_ceil_mode() {
        // Test ceil_mode=true produces larger output when input doesn't divide evenly by stride
        // Input: 1x1x6 (values 0-5), kernel: 3, stride: 2, padding: 0
        // Floor mode: output = (6-3)/2+1 = 2 elements
        // Ceil mode: output = ceil((6-3)/2)+1 = ceil(1.5)+1 = 3 elements
        let x = TestTensor::from([[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]]);

        // With ceil_mode=false (floor): output is 2 elements
        // Window 0: avg(0,1,2) = 1
        // Window 1: avg(2,3,4) = 3
        let y_floor = TestTensor::<3>::from([[[1.0, 3.0]]]);

        let output_floor = avg_pool1d(
            x.clone(),
            3,    // kernel_size
            2,    // stride
            0,    // padding
            true, // count_include_pad
            false,
        );

        y_floor.to_data().assert_approx_eq::<FT>(
            &output_floor.into_data(),
            Tolerance::default().set_half_precision_relative(1e-3),
        );

        // With ceil_mode=true: output is 3 elements
        // Window 0: avg(0,1,2) = 1
        // Window 1: avg(2,3,4) = 3
        // Window 2: avg(4,5) = 4.5 (partial window, count_include_pad=false divides by 2)
        let y_ceil = TestTensor::<3>::from([[[1.0, 3.0, 4.5]]]);

        let output_ceil = avg_pool1d(
            x, 3,     // kernel_size
            2,     // stride
            0,     // padding
            false, // count_include_pad=false to get correct average for partial window
            true,
        );

        y_ceil.to_data().assert_approx_eq::<FT>(
            &output_ceil.into_data(),
            Tolerance::default().set_half_precision_relative(1e-3),
        );
    }

    #[test]
    fn test_avg_pool1d_ceil_mode_count_include_pad() {
        // Test count_include_pad=true + ceil_mode=true interaction for 1D
        // When ceil_mode creates windows that extend beyond the padded input:
        // - count_include_pad=true should count positions within padded bounds (not ceil_mode extensions)
        //
        // Input: 1x1x6, kernel 3, stride 2, padding 1, ceil_mode=true
        // Output is 4 elements
        let x = TestTensor::from([[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]]);

        // Expected PyTorch output with padding=1, ceil_mode=true, count_include_pad=true:
        // Window 0: positions -1,0,1 -> values 0,0,1 (0 is padding) / 3 = 0.333
        // Window 1: positions 1,2,3 -> values 1,2,3 / 3 = 2.0
        // Window 2: positions 3,4,5 -> values 3,4,5 / 3 = 4.0
        // Window 3: positions 5,6,7 -> only 5 is valid, 6 is padding, 7 is ceil_mode extension
        //           value 5 / 2 (only 2 positions within padded bounds) = 2.5
        let expected = TestTensor::<3>::from([[[0.3333, 2.0, 4.0, 2.5]]]);

        let output = avg_pool1d(
            x, 3,    // kernel_size
            2,    // stride
            1,    // padding
            true, // count_include_pad=true
            true, // ceil_mode=true
        );

        expected.to_data().assert_approx_eq::<FT>(
            &output.into_data(),
            Tolerance::default().set_half_precision_relative(1e-2),
        );
    }
}
