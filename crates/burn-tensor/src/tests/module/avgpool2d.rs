#[burn_tensor_testgen::testgen(module_avg_pool2d)]
mod tests {
    use super::*;
    use burn_tensor::module::avg_pool2d;
    use burn_tensor::{Shape, Tensor};
    use burn_tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn test_avg_pool2d_simple() {
        let test = AvgPool2dTestCase {
            batch_size: 1,
            channels: 1,
            kernel_size_1: 3,
            kernel_size_2: 3,
            padding_1: 0,
            padding_2: 0,
            stride_1: 1,
            stride_2: 1,
            height: 6,
            width: 6,
            count_include_pad: true,
        };

        test.assert_output(TestTensor::from([[[
            [7., 8., 9., 10.],
            [13., 14., 15., 16.],
            [19., 20., 21., 22.],
            [25., 26., 27., 28.],
        ]]]));
    }

    #[test]
    fn test_avg_pool2d_complex() {
        let test = AvgPool2dTestCase {
            batch_size: 1,
            channels: 1,
            kernel_size_1: 3,
            kernel_size_2: 4,
            padding_1: 1,
            padding_2: 2,
            stride_1: 1,
            stride_2: 2,
            height: 4,
            width: 6,
            count_include_pad: true,
        };

        test.assert_output(TestTensor::from([[[
            [1.1667, 3.0000, 4.3333, 2.5000],
            [3.2500, 7.5000, 9.5000, 5.2500],
            [6.2500, 13.5000, 15.5000, 8.2500],
            [5.1667, 11.0000, 12.3333, 6.5000],
        ]]]));
    }

    #[test]
    fn test_avg_pool2d_complex_dont_include_pad() {
        let test = AvgPool2dTestCase {
            batch_size: 1,
            channels: 1,
            kernel_size_1: 3,
            kernel_size_2: 4,
            padding_1: 1,
            padding_2: 2,
            stride_1: 1,
            stride_2: 2,
            height: 4,
            width: 6,
            count_include_pad: false,
        };

        test.assert_output(TestTensor::from([[[
            [3.5000, 4.5000, 6.5000, 7.5000],
            [6.5000, 7.5000, 9.5000, 10.5000],
            [12.5000, 13.5000, 15.5000, 16.5000],
            [15.5000, 16.5000, 18.5000, 19.5000],
        ]]]));
    }

    struct AvgPool2dTestCase {
        batch_size: usize,
        channels: usize,
        kernel_size_1: usize,
        kernel_size_2: usize,
        padding_1: usize,
        padding_2: usize,
        stride_1: usize,
        stride_2: usize,
        height: usize,
        width: usize,
        count_include_pad: bool,
    }

    impl AvgPool2dTestCase {
        fn assert_output(self, y: TestTensor<4>) {
            let shape_x = Shape::new([self.batch_size, self.channels, self.height, self.width]);
            let x = TestTensor::from(
                TestTensorInt::arange(0..shape_x.num_elements() as i64, &y.device())
                    .reshape::<4, _>(shape_x)
                    .into_data(),
            );
            let output = avg_pool2d(
                x,
                [self.kernel_size_1, self.kernel_size_2],
                [self.stride_1, self.stride_2],
                [self.padding_1, self.padding_2],
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
    fn test_avg_pool2d_ceil_mode() {
        // Test ceil_mode=true produces larger output when input doesn't divide evenly by stride
        // Input: 1x1x6x6 (values 0-35), kernel: 3x3, stride: 2x2, padding: 0x0
        // Floor mode: output = (6-3)/2+1 = 2 x 2
        // Ceil mode: output = ceil((6-3)/2)+1 = ceil(1.5)+1 = 3 x 3
        let x = TestTensor::from([[[
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
            [18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
            [24.0, 25.0, 26.0, 27.0, 28.0, 29.0],
            [30.0, 31.0, 32.0, 33.0, 34.0, 35.0],
        ]]]);

        // With ceil_mode=false (floor): output is 2x2
        // Window (0,0): avg(0,1,2,6,7,8,12,13,14) = avg(63) = 7
        // Window (0,1): avg(2,3,4,8,9,10,14,15,16) = avg(81) = 9
        // Window (1,0): avg(12,13,14,18,19,20,24,25,26) = avg(171) = 19
        // Window (1,1): avg(14,15,16,20,21,22,26,27,28) = avg(189) = 21
        let y_floor = TestTensor::<4>::from([[[[7.0, 9.0], [19.0, 21.0]]]]);

        let output_floor = avg_pool2d(
            x.clone(),
            [3, 3],
            [2, 2],
            [0, 0],
            true, // count_include_pad
            false,
        );

        y_floor.to_data().assert_approx_eq::<FT>(
            &output_floor.into_data(),
            Tolerance::default().set_half_precision_relative(1e-3),
        );

        // With ceil_mode=true: output is 3x3
        // The extra windows at the edge include partial/padded regions
        // When count_include_pad=false, only actual values are averaged
        // Window (0,2): positions (0:3, 4:6) -> values 4,5,10,11,16,17 -> avg = 10.5
        // Window (1,2): positions (2:5, 4:6) -> values 16,17,22,23,28,29 -> avg = 22.5
        // Window (2,0): positions (4:6, 0:3) -> values 24,25,26,30,31,32 -> avg = 28
        // Window (2,1): positions (4:6, 2:5) -> values 26,27,28,32,33,34 -> avg = 30
        // Window (2,2): positions (4:6, 4:6) -> values 28,29,34,35 -> avg = 31.5
        let y_ceil =
            TestTensor::<4>::from([[[[7.0, 9.0, 10.5], [19.0, 21.0, 22.5], [28.0, 30.0, 31.5]]]]);

        let output_ceil = avg_pool2d(
            x,
            [3, 3],
            [2, 2],
            [0, 0],
            false, // count_include_pad=false to avoid dividing by full kernel size
            true,
        );

        y_ceil.to_data().assert_approx_eq::<FT>(
            &output_ceil.into_data(),
            Tolerance::default().set_half_precision_relative(1e-3),
        );
    }

    #[test]
    fn test_avg_pool2d_ceil_mode_count_include_pad() {
        // Test count_include_pad=true + ceil_mode=true interaction
        // When ceil_mode creates windows that extend beyond the padded input:
        // - count_include_pad=true should count positions within padded bounds (not ceil_mode extensions)
        //
        // For input 6x6, kernel 3, stride 2, padding 1, ceil_mode=true:
        // - Output is 4x4
        // - Corner (3,3) window covers positions beyond even the user padding
        // - Expected: 35/4 = 8.75 (divides by count of positions within padded bounds)

        let x = TestTensor::from([[[
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
            [18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
            [24.0, 25.0, 26.0, 27.0, 28.0, 29.0],
            [30.0, 31.0, 32.0, 33.0, 34.0, 35.0],
        ]]]);

        // Expected PyTorch output with padding=1, ceil_mode=true, count_include_pad=true
        // Note: corner (3,3) = 8.75 = 35/4, not 35/9
        let expected = TestTensor::<4>::from([[[
            [1.5556, 3.3333, 4.6667, 2.6667],
            [8.3333, 14.0000, 16.0000, 8.5000],
            [16.3333, 26.0000, 28.0000, 14.5000],
            [10.1667, 16.0000, 17.0000, 8.7500],
        ]]]);

        let output = avg_pool2d(
            x,
            [3, 3],
            [2, 2],
            [1, 1],
            true, // count_include_pad=true
            true, // ceil_mode=true
        );

        expected.to_data().assert_approx_eq::<FT>(
            &output.into_data(),
            Tolerance::default().set_half_precision_relative(1e-2),
        );
    }
}
