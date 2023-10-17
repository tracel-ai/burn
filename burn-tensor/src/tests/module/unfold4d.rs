#[burn_tensor_testgen::testgen(module_unfold4d)]
mod tests {
    use super::*;
    use burn_tensor::module::unfold4d;
    use burn_tensor::ops::UnfoldOptions;
    use burn_tensor::{Data, Shape, Tensor};

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
            let x = TestTensor::from_data(
                TestTensorInt::arange(0..shape_x.num_elements())
                    .reshape(shape_x)
                    .into_data()
                    .convert(),
            );

            let output = unfold4d(
                x,
                self.kernel_size,
                UnfoldOptions::new(Some(self.stride), Some(self.padding), Some(self.dilation)),
            );

            assert_eq!(
                output.shape().dims,
                expected_shape,
                "Expected shape doesn't match the actual shape"
            );
        }
    }
}
