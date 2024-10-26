use crate::{
    kernel::{
        self,
        conv::{Conv2dStrategy, ConvTranspose2dStrategy},
    },
    FloatElement, IntElement, JitBackend, JitRuntime,
};
use burn_tensor::ops::{
    ConvOptions, ConvTransposeOptions, DeformConv2dBackward, DeformConvOptions, InterpolateOptions,
    MaxPool2dBackward, MaxPool2dWithIndices, ModuleOps,
};
use burn_tensor::ops::{FloatTensor, IntTensor};

impl<R, F, I> ModuleOps<Self> for JitBackend<R, F, I>
where
    R: JitRuntime,
    F: FloatElement,
    I: IntElement,
{
    fn conv2d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<2>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv2d::<R, F, I>(x, weight, bias, options, Conv2dStrategy::default())
    }

    fn deform_conv2d(
        x: FloatTensor<Self>,
        offset: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        mask: Option<FloatTensor<Self>>,
        bias: Option<FloatTensor<Self>>,
        options: DeformConvOptions<2>,
    ) -> FloatTensor<Self> {
        kernel::conv::deform_conv2d::<R, F, I>(x, offset, weight, mask, bias, options)
    }

    fn deform_conv2d_backward(
        x: FloatTensor<Self>,
        offset: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        mask: Option<FloatTensor<Self>>,
        bias: Option<FloatTensor<Self>>,
        output_grad: FloatTensor<Self>,
        options: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<Self> {
        kernel::conv::deform_conv2d_backward(x, offset, weight, mask, bias, output_grad, options)
    }

    fn conv3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<3>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv3d(x, weight, bias, options)
    }

    fn conv_transpose2d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv_transpose2d::<R, F, I>(
            x,
            weight,
            bias,
            options,
            ConvTranspose2dStrategy::default(),
        )
    }

    fn conv_transpose3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<3>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv_transpose3d(x, weight, bias, options)
    }

    fn avg_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> FloatTensor<Self> {
        kernel::pool::avg_pool2d(x, kernel_size, stride, padding, count_include_pad)
    }

    fn avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> FloatTensor<Self> {
        kernel::pool::avg_pool2d_backward(x, grad, kernel_size, stride, padding, count_include_pad)
    }

    fn max_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> FloatTensor<Self> {
        kernel::pool::max_pool2d(x, kernel_size, stride, padding, dilation)
    }

    fn max_pool2d_with_indices(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> MaxPool2dWithIndices<Self> {
        let (output, indices) =
            kernel::pool::max_pool2d_with_indices(x, kernel_size, stride, padding, dilation);

        MaxPool2dWithIndices::new(output, indices)
    }

    fn max_pool2d_with_indices_backward(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        output_grad: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> MaxPool2dBackward<Self> {
        MaxPool2dBackward::new(kernel::pool::max_pool2d_with_indices_backward(
            x,
            output_grad,
            indices,
            kernel_size,
            stride,
            padding,
            dilation,
        ))
    }

    fn adaptive_avg_pool2d(x: FloatTensor<Self>, output_size: [usize; 2]) -> FloatTensor<Self> {
        kernel::pool::adaptive_avg_pool2d(x, output_size)
    }

    fn adaptive_avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        kernel::pool::adaptive_avg_pool2d_backward(x, grad)
    }

    fn interpolate(
        x: FloatTensor<Self>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Self> {
        kernel::interpolate::interpolate(x, output_size, options)
    }

    fn interpolate_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Self> {
        kernel::interpolate::interpolate_backward(x, grad, output_size, options)
    }
}
