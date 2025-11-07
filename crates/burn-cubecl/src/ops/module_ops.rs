use crate::{
    CubeBackend, CubeRuntime, FloatElement, IntElement,
    element::BoolElement,
    execute_with_dtype,
    kernel::{
        self,
        conv::{ConvStrategy, ConvTranspose2dStrategy},
    },
};
use burn_tensor::ops::{
    ConvOptions, ConvTransposeOptions, DeformConv2dBackward, DeformConvOptions, InterpolateOptions,
    MaxPool2dBackward, MaxPool2dWithIndices, ModuleOps,
};
use burn_tensor::ops::{FloatTensor, IntTensor};

impl<R, F, I, BT> ModuleOps<Self> for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn conv1d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<1>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv::<R, 1>(x, weight, bias, options, ConvStrategy::default()).unwrap()
    }

    fn conv2d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<2>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv::<R, 2>(x, weight, bias, options, ConvStrategy::default()).unwrap()
    }

    fn deform_conv2d(
        x: FloatTensor<Self>,
        offset: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        mask: Option<FloatTensor<Self>>,
        bias: Option<FloatTensor<Self>>,
        options: DeformConvOptions<2>,
    ) -> FloatTensor<Self> {
        execute_with_dtype!(
            float(x.dtype),
            E,
            kernel::conv::deform_conv2d::<R, E>(x, offset, weight, mask, bias, options).unwrap()
        )
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
        execute_with_dtype!(float(x.dtype), E, {
            let (x, o, w, m, b) = kernel::conv::deform_conv2d_backward::<R, E, I, BT>(
                x,
                offset,
                weight,
                mask,
                bias,
                output_grad,
                options,
            )
            .unwrap();
            DeformConv2dBackward::new(x, o, w, m, b)
        })
    }

    fn conv3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<3>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv::<R, 3>(x, weight, bias, options, ConvStrategy::Direct).unwrap()
    }

    fn conv_transpose2d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<Self> {
        execute_with_dtype!(
            float(x.dtype),
            E,
            kernel::conv::conv_transpose2d::<R, E, I>(
                x,
                weight,
                bias,
                options,
                ConvTranspose2dStrategy::default(),
            )
            .unwrap()
        )
    }

    fn conv_transpose3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<3>,
    ) -> FloatTensor<Self> {
        execute_with_dtype!(
            float(x.dtype),
            E,
            kernel::conv::conv_transpose3d::<R, E>(x, weight, bias, options)
        )
    }

    fn avg_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> FloatTensor<Self> {
        execute_with_dtype!(
            float(x.dtype),
            E,
            kernel::pool::avg_pool2d::<R, E>(x, kernel_size, stride, padding, count_include_pad)
        )
    }

    fn avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> FloatTensor<Self> {
        execute_with_dtype!(
            float(x.dtype),
            E,
            kernel::pool::avg_pool2d_backward::<R, E>(
                x,
                grad,
                kernel_size,
                stride,
                padding,
                count_include_pad,
            )
        )
    }

    fn max_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> FloatTensor<Self> {
        execute_with_dtype!(
            float(x.dtype),
            E,
            kernel::pool::max_pool2d::<R, E>(x, kernel_size, stride, padding, dilation)
        )
    }

    fn max_pool2d_with_indices(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> MaxPool2dWithIndices<Self> {
        execute_with_dtype!(float(x.dtype), E, {
            let (output, indices) = kernel::pool::max_pool2d_with_indices::<R, E, I>(
                x,
                kernel_size,
                stride,
                padding,
                dilation,
            );

            MaxPool2dWithIndices::new(output, indices)
        })
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
        execute_with_dtype!(
            int(indices.dtype),
            I,
            execute_with_dtype!(
                float(x.dtype),
                E,
                MaxPool2dBackward::new(kernel::pool::max_pool2d_with_indices_backward::<R, E, I>(
                    x,
                    output_grad,
                    indices,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                ))
            )
        )
    }

    fn adaptive_avg_pool2d(x: FloatTensor<Self>, output_size: [usize; 2]) -> FloatTensor<Self> {
        execute_with_dtype!(
            float(x.dtype),
            E,
            kernel::pool::adaptive_avg_pool2d::<R, E>(x, output_size)
        )
    }

    fn adaptive_avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        execute_with_dtype!(
            float(x.dtype),
            E,
            kernel::pool::adaptive_avg_pool2d_backward::<R, E>(x, grad)
        )
    }

    fn interpolate(
        x: FloatTensor<Self>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Self> {
        execute_with_dtype!(
            float(x.dtype),
            E,
            kernel::interpolate::interpolate::<R, E>(x, output_size, options)
        )
    }

    fn interpolate_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Self> {
        execute_with_dtype!(
            float(x.dtype),
            E,
            kernel::interpolate::interpolate_backward::<R, E>(x, grad, output_size, options)
        )
    }
}
