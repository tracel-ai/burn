#[burn_tensor_testgen::testgen(module_unfold4d)]
mod tests {
    use super::*;
    use burn_tensor::module::unfold4d;
    use burn_tensor::ops::UnfoldOptions;
    use burn_tensor::{Shape, Tensor};

    #[test]
    fn test_unfold4d_shape() {
        let test = Unfold4dTestCase {
            batch_size: 2,
            channels_in: 5,
            kernel_size: [2, 3],
            padding: [0, 0],
            stride: [1, 1],
            dilation: [1, 1],
            height: 3,
            width: 4,
        };

        test.assert_shape([2, 30, 4]);
    }

    #[test]
    fn test_unfold4d_simple() {
        let test = Unfold4dTestCase {
            batch_size: 1,
            channels_in: 2,
            kernel_size: [2, 2],
            padding: [0, 0],
            stride: [1, 1],
            dilation: [1, 1],
            height: 4,
            width: 4,
        };

        test.assert_output(TestTensor::from([[
            [0., 1., 2., 4., 5., 6., 8., 9., 10.],
            [1., 2., 3., 5., 6., 7., 9., 10., 11.],
            [4., 5., 6., 8., 9., 10., 12., 13., 14.],
            [5., 6., 7., 9., 10., 11., 13., 14., 15.],
            [16., 17., 18., 20., 21., 22., 24., 25., 26.],
            [17., 18., 19., 21., 22., 23., 25., 26., 27.],
            [20., 21., 22., 24., 25., 26., 28., 29., 30.],
            [21., 22., 23., 25., 26., 27., 29., 30., 31.],
        ]]));
    }

    #[test]
    fn test_unfold4d_complex() {
        let test = Unfold4dTestCase {
            batch_size: 1,
            channels_in: 2,
            kernel_size: [2, 3],
            padding: [0, 1],
            stride: [1, 2],
            dilation: [1, 2],
            height: 3,
            width: 4,
        };

        test.assert_output(TestTensor::from([[
            [0., 0.],
            [1., 5.],
            [3., 7.],
            [0., 0.],
            [5., 9.],
            [7., 11.],
            [0., 0.],
            [13., 17.],
            [15., 19.],
            [0., 0.],
            [17., 21.],
            [19., 23.],
        ]]));
    }

    struct Unfold4dTestCase {
        batch_size: usize,
        channels_in: usize,
        kernel_size: [usize; 2],
        padding: [usize; 2],
        stride: [usize; 2],
        dilation: [usize; 2],
        height: usize,
        width: usize,
    }

    impl Unfold4dTestCase {
        fn assert_shape(self, expected_shape: [usize; 3]) {
            let shape_x = Shape::new([self.batch_size, self.channels_in, self.height, self.width]);
            let x = TestTensor::from(
                TestTensorInt::arange(0..shape_x.num_elements() as i64, &Default::default())
                    .reshape::<4, _>(shape_x)
                    .into_data()
                    .convert::<f32>(),
            );

            let output = unfold4d(
                x,
                self.kernel_size,
                UnfoldOptions::new(self.stride, self.padding, self.dilation),
            );

            assert_eq!(
                output.shape().dims,
                expected_shape,
                "Expected shape doesn't match the actual shape"
            );
        }

        fn assert_output(self, expected: TestTensor<3>) {
            let shape_x = Shape::new([self.batch_size, self.channels_in, self.height, self.width]);
            let x = TestTensor::from(
                TestTensorInt::arange(0..shape_x.num_elements() as i64, &Default::default())
                    .reshape::<4, _>(shape_x)
                    .into_data(),
            );

            let output = unfold4d(
                x,
                self.kernel_size,
                UnfoldOptions::new(self.stride, self.padding, self.dilation),
            );

            output
                .into_data()
                .assert_approx_eq(&expected.into_data(), 3);
        }
    }
}
