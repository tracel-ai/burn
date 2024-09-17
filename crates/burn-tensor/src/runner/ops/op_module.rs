use crate::ops::{
    IntTensor, InterpolateOptions, MaxPool1dBackward, MaxPool1dWithIndices, MaxPool2dBackward,
    MaxPool2dWithIndices,
};
use crate::runner::Runner;
use crate::{
    ops::{ConvOptions, ConvTransposeOptions, FloatTensor, ModuleOps},
    runner::RunnerBackend,
};

impl<B: RunnerBackend> ModuleOps<Self> for Runner<B> {
    fn conv1d(
        x: FloatTensor<Self, 3>,
        weight: FloatTensor<Self, 3>,
        bias: Option<FloatTensor<Self, 1>>,
        options: ConvOptions<1>,
    ) -> FloatTensor<Self, 3> {
        todo!()
    }

    fn conv2d(
        x: FloatTensor<Self, 4>,
        weight: FloatTensor<Self, 4>,
        bias: Option<FloatTensor<Self, 1>>,
        options: ConvOptions<2>,
    ) -> FloatTensor<Self, 4> {
        todo!()
    }

    fn conv3d(
        x: FloatTensor<Self, 5>,
        weight: FloatTensor<Self, 5>,
        bias: Option<FloatTensor<Self, 1>>,
        options: ConvOptions<3>,
    ) -> FloatTensor<Self, 5> {
        todo!()
    }

    fn conv_transpose1d(
        x: FloatTensor<Self, 3>,
        weight: FloatTensor<Self, 3>,
        bias: Option<FloatTensor<Self, 1>>,
        options: ConvTransposeOptions<1>,
    ) -> FloatTensor<Self, 3> {
        todo!()
    }

    fn conv_transpose2d(
        x: FloatTensor<Self, 4>,
        weight: FloatTensor<Self, 4>,
        bias: Option<FloatTensor<Self, 1>>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<Self, 4> {
        todo!()
    }

    fn conv_transpose3d(
        x: FloatTensor<Self, 5>,
        weight: FloatTensor<Self, 5>,
        bias: Option<FloatTensor<Self, 1>>,
        options: ConvTransposeOptions<3>,
    ) -> FloatTensor<Self, 5> {
        todo!()
    }

    fn avg_pool1d(
        x: FloatTensor<Self, 3>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
    ) -> FloatTensor<Self, 3> {
        todo!()
    }

    fn avg_pool2d(
        x: FloatTensor<Self, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> FloatTensor<Self, 4> {
        todo!()
    }

    fn avg_pool1d_backward(
        x: FloatTensor<Self, 3>,
        grad: FloatTensor<Self, 3>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
    ) -> FloatTensor<Self, 3> {
        todo!()
    }

    fn avg_pool2d_backward(
        x: FloatTensor<Self, 4>,
        grad: FloatTensor<Self, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> FloatTensor<Self, 4> {
        todo!()
    }

    fn max_pool1d(
        x: FloatTensor<Self, 3>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> FloatTensor<Self, 3> {
        todo!()
    }

    fn max_pool2d(
        x: FloatTensor<Self, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> FloatTensor<Self, 4> {
        todo!()
    }

    fn max_pool1d_with_indices(
        x: FloatTensor<Self, 3>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> MaxPool1dWithIndices<Self> {
        todo!()
    }

    fn max_pool2d_with_indices(
        x: FloatTensor<Self, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> MaxPool2dWithIndices<Self> {
        todo!()
    }

    fn max_pool1d_with_indices_backward(
        x: FloatTensor<Self, 3>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        output_grad: FloatTensor<Self, 3>,
        indices: IntTensor<Self, 3>,
    ) -> MaxPool1dBackward<Self> {
        todo!()
    }

    fn max_pool2d_with_indices_backward(
        x: FloatTensor<Self, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        output_grad: FloatTensor<Self, 4>,
        indices: IntTensor<Self, 4>,
    ) -> MaxPool2dBackward<Self> {
        todo!()
    }

    fn adaptive_avg_pool1d(x: FloatTensor<Self, 3>, output_size: usize) -> FloatTensor<Self, 3> {
        todo!()
    }

    fn adaptive_avg_pool2d(
        x: FloatTensor<Self, 4>,
        output_size: [usize; 2],
    ) -> FloatTensor<Self, 4> {
        todo!()
    }

    fn adaptive_avg_pool1d_backward(
        x: FloatTensor<Self, 3>,
        grad: FloatTensor<Self, 3>,
    ) -> FloatTensor<Self, 3> {
        todo!()
    }

    fn adaptive_avg_pool2d_backward(
        x: FloatTensor<Self, 4>,
        grad: FloatTensor<Self, 4>,
    ) -> FloatTensor<Self, 4> {
        todo!()
    }

    fn interpolate(
        x: FloatTensor<Self, 4>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Self, 4> {
        todo!()
    }

    fn interpolate_backward(
        x: FloatTensor<Self, 4>,
        grad: FloatTensor<Self, 4>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Self, 4> {
        todo!()
    }
}
