#[burn_tensor_testgen::testgen(module_conv1d)]
mod tests {
    use super::*;
    use burn_tensor::module::conv1d;
    use burn_tensor::ops::ConvOptions;
    use burn_tensor::{Shape, Tensor};

    #[test]
    fn test_conv1d_simple() {
        let test = Conv1dTestCase {
            batch_size: 2,
            channels_in: 2,
            channels_out: 2,
            kernel_size: 3,
            padding: 1,
            stride: 1,
            dilation: 1,
            groups: 1,
            length: 4,
        };

        test.assert_output(TestTensor::from([
            [[43., 67., 82., 49.], [104., 176., 227., 158.]],
            [[139., 187., 202., 113.], [392., 584., 635., 414.]],
        ]));
    }

    #[test]
    fn test_conv1d_dilation() {
        let test = Conv1dTestCase {
            batch_size: 2,
            channels_in: 2,
            channels_out: 2,
            kernel_size: 3,
            padding: 1,
            stride: 1,
            dilation: 2,
            groups: 1,
            length: 4,
        };

        test.assert_output(TestTensor::from([
            [[62., 38.], [159., 111.]],
            [[158., 102.], [447., 367.]],
        ]));
    }

    #[test]
    fn test_conv1d_groups() {
        let test = Conv1dTestCase {
            batch_size: 2,
            channels_in: 2,
            channels_out: 2,
            kernel_size: 3,
            padding: 1,
            stride: 1,
            dilation: 1,
            groups: 2,
            length: 4,
        };

        test.assert_output(TestTensor::from([
            [[2., 5., 8., 3.], [42., 63., 75., 47.]],
            [[26., 29., 32., 11.], [114., 159., 171., 103.]],
        ]));
    }

    #[test]
    fn test_conv1d_complex() {
        let test = Conv1dTestCase {
            batch_size: 2,
            channels_in: 3,
            channels_out: 4,
            kernel_size: 3,
            padding: 1,
            stride: 2,
            dilation: 1,
            groups: 1,
            length: 4,
        };

        test.assert_output(TestTensor::from_floats(
            [
                [[171., 294.], [415., 781.], [659., 1268.], [903., 1755.]],
                [[495., 726.], [1387., 2185.], [2279., 3644.], [3171., 5103.]],
            ],
            &Default::default(),
        ));
    }

    struct Conv1dTestCase {
        batch_size: usize,
        channels_in: usize,
        channels_out: usize,
        kernel_size: usize,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        length: usize,
    }

    impl Conv1dTestCase {
        fn assert_output(self, y: TestTensor<3>) {
            let shape_x = Shape::new([self.batch_size, self.channels_in, self.length]);
            let shape_weight = Shape::new([
                self.channels_out,
                self.channels_in / self.groups,
                self.kernel_size,
            ]);
            let device = Default::default();
            let weight = TestTensor::from_data(
                TestTensorInt::arange(0..shape_weight.num_elements() as i64, &device)
                    .reshape::<3, _>(shape_weight)
                    .into_data(),
                &device,
            );
            let bias = TestTensor::from_data(
                TestTensorInt::arange(0..self.channels_out as i64, &device).into_data(),
                &device,
            );
            let x = TestTensor::from_data(
                TestTensorInt::arange(0..shape_x.num_elements() as i64, &device)
                    .reshape::<3, _>(shape_x)
                    .into_data(),
                &device,
            );
            let output = conv1d(
                x,
                weight,
                Some(bias),
                ConvOptions::new([self.stride], [self.padding], [self.dilation], self.groups),
            );

            y.to_data().assert_approx_eq(&output.into_data(), 3);
        }
    }
}
