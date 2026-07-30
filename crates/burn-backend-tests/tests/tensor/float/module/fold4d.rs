use super::*;
use burn_tensor::Shape;
use burn_tensor::Tolerance;
use burn_tensor::module::fold4d;
use burn_tensor::ops::UnfoldOptions;

#[test]
fn test_fold4d_shape() {
    let test = Fold4dTestCase {
        batch_size: 2,
        channels: 3,
        kernel_size: [2, 2],
        padding: [1, 1],
        stride: [2, 2],
        dilation: [1, 1],
        output_size: [3, 3],
    };

    test.assert_shape([2, 3, 3, 3]);
}

#[test]
fn test_fold4d_stride_padding() {
    // Non-unit stride (2) and non-zero padding (1); reference from torch.nn.functional.fold.
    let test = Fold4dTestCase {
        batch_size: 1,
        channels: 1,
        kernel_size: [2, 2],
        padding: [1, 1],
        stride: [2, 2],
        dilation: [1, 1],
        output_size: [3, 3],
    };

    test.assert_output(TestTensor::from([[[
        [12., 9., 13.],
        [6., 3., 7.],
        [14., 11., 15.],
    ]]]));
}

#[test]
fn test_fold4d_stride_no_padding() {
    // Non-unit stride (2), no padding (perfect tiling); reference from torch.nn.functional.fold.
    let test = Fold4dTestCase {
        batch_size: 1,
        channels: 1,
        kernel_size: [2, 2],
        padding: [0, 0],
        stride: [2, 2],
        dilation: [1, 1],
        output_size: [4, 4],
    };

    test.assert_output(TestTensor::from([[[
        [0., 4., 1., 5.],
        [8., 12., 9., 13.],
        [2., 6., 3., 7.],
        [10., 14., 11., 15.],
    ]]]));
}

struct Fold4dTestCase {
    batch_size: usize,
    channels: usize,
    kernel_size: [usize; 2],
    padding: [usize; 2],
    stride: [usize; 2],
    dilation: [usize; 2],
    output_size: [usize; 2],
}

impl Fold4dTestCase {
    fn num_blocks(&self) -> usize {
        let blocks_height = (self.output_size[0] + 2 * self.padding[0]
            - self.dilation[0] * (self.kernel_size[0] - 1)
            - 1)
            / self.stride[0]
            + 1;
        let blocks_width = (self.output_size[1] + 2 * self.padding[1]
            - self.dilation[1] * (self.kernel_size[1] - 1)
            - 1)
            / self.stride[1]
            + 1;
        blocks_height * blocks_width
    }

    fn columns(&self) -> TestTensor<3> {
        let channels_col = self.channels * self.kernel_size[0] * self.kernel_size[1];
        let shape_x = Shape::new([self.batch_size, channels_col, self.num_blocks()]);
        TestTensor::from(
            TestTensorInt::arange(0..shape_x.num_elements() as i64, &Default::default())
                .reshape::<3, _>(shape_x)
                .into_data(),
        )
    }

    fn assert_shape(self, expected_shape: [usize; 4]) {
        let output = fold4d(
            self.columns(),
            self.output_size,
            self.kernel_size,
            UnfoldOptions::new(self.stride, self.padding, self.dilation),
        );

        assert_eq!(
            output.shape().as_slice(),
            expected_shape,
            "Expected shape doesn't match the actual shape"
        );
    }

    fn assert_output(self, expected: TestTensor<4>) {
        let output = fold4d(
            self.columns(),
            self.output_size,
            self.kernel_size,
            UnfoldOptions::new(self.stride, self.padding, self.dilation),
        );

        let tolerance = Tolerance::default()
            .set_half_precision_relative(2e-3)
            .set_half_precision_absolute(2e-3);
        output
            .into_data()
            .assert_approx_eq::<FloatElem>(&expected.into_data(), tolerance);
    }
}
