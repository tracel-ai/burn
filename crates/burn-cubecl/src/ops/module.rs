use crate::{
    CubeBackend, CubeRuntime, FloatElement, IntElement,
    element::BoolElement,
    kernel::{self, conv::ConvTranspose2dStrategy},
};
use burn_backend::ops::{
    ConvOptions, ConvTransposeOptions, DeformConv2dBackward, DeformConvOptions, InterpolateOptions,
    MaxPool2dBackward, MaxPool2dWithIndices, MaxPool3dBackward, MaxPool3dWithIndices, ModuleOps,
};
use burn_backend::tensor::{BoolTensor, FloatTensor, IntTensor};

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
        kernel::conv::conv_forward::<R, 1>(x, weight, bias, options, Default::default()).unwrap()
    }

    fn conv1d_x_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: ConvOptions<1>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv_data_backward(output_grad, weight, x.shape, options, Default::default())
            .unwrap()
    }

    fn conv1d_weight_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: ConvOptions<1>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv_weight_backward::<R, 1>(
            x,
            output_grad,
            weight.shape.clone(),
            options,
            Default::default(),
        )
        .unwrap()
    }

    fn conv2d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<2>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv_forward::<R, 2>(x, weight, bias, options, Default::default()).unwrap()
    }

    fn conv2d_x_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: ConvOptions<2>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv_data_backward(output_grad, weight, x.shape, options, Default::default())
            .unwrap()
    }

    fn conv2d_weight_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: ConvOptions<2>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv_weight_backward::<R, 2>(
            x,
            output_grad,
            weight.shape.clone(),
            options,
            Default::default(),
        )
        .unwrap()
    }

    fn deform_conv2d(
        x: FloatTensor<Self>,
        offset: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        mask: Option<FloatTensor<Self>>,
        bias: Option<FloatTensor<Self>>,
        options: DeformConvOptions<2>,
    ) -> FloatTensor<Self> {
        kernel::conv::deform_conv2d(x, offset, weight, mask, bias, options).unwrap()
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
        let (x, o, w, m, b) = kernel::conv::deform_conv2d_backward(
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
    }

    fn conv3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<3>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv_forward::<R, 3>(x, weight, bias, options, Default::default()).unwrap()
    }

    fn conv3d_x_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: ConvOptions<3>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv_data_backward(output_grad, weight, x.shape, options, Default::default())
            .unwrap()
    }

    fn conv3d_weight_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: ConvOptions<3>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv_weight_backward::<R, 3>(
            x,
            output_grad,
            weight.shape.clone(),
            options,
            Default::default(),
        )
        .unwrap()
    }

    fn conv_transpose2d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv_transpose2d(x, weight, bias, options, ConvTranspose2dStrategy::default())
            .unwrap()
    }

    fn conv_transpose3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<3>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv_transpose3d(x, weight, bias, options).expect("Kernel to never fail")
    }

    fn avg_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        kernel::pool::avg_pool2d(
            x,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        )
    }

    fn avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        kernel::pool::avg_pool2d_backward(
            x,
            grad,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        )
    }

    fn max_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        kernel::pool::max_pool2d(x, kernel_size, stride, padding, dilation, ceil_mode)
    }

    fn max_pool2d_with_indices(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> MaxPool2dWithIndices<Self> {
        let (output, indices) = kernel::pool::max_pool2d_with_indices(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            I::dtype(),
        );

        MaxPool2dWithIndices::new(output, indices)
    }

    fn max_pool2d_with_indices_backward(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
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
            ceil_mode,
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

    fn avg_pool3d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 3],
        stride: [usize; 3],
        padding: [usize; 3],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        kernel::pool::avg_pool3d(
            x,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        )
    }

    fn avg_pool3d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        kernel_size: [usize; 3],
        stride: [usize; 3],
        padding: [usize; 3],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        kernel::pool::avg_pool3d_backward(
            x,
            grad,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        )
    }

    fn max_pool3d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 3],
        stride: [usize; 3],
        padding: [usize; 3],
        dilation: [usize; 3],
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        kernel::pool::max_pool3d(x, kernel_size, stride, padding, dilation, ceil_mode)
    }

    fn max_pool3d_with_indices(
        x: FloatTensor<Self>,
        kernel_size: [usize; 3],
        stride: [usize; 3],
        padding: [usize; 3],
        dilation: [usize; 3],
        ceil_mode: bool,
    ) -> MaxPool3dWithIndices<Self> {
        let (output, indices) = kernel::pool::max_pool3d_with_indices(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            I::dtype(),
        );

        MaxPool3dWithIndices::new(output, indices)
    }

    fn max_pool3d_with_indices_backward(
        x: FloatTensor<Self>,
        kernel_size: [usize; 3],
        stride: [usize; 3],
        padding: [usize; 3],
        dilation: [usize; 3],
        ceil_mode: bool,
        output_grad: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> MaxPool3dBackward<Self> {
        MaxPool3dBackward::new(kernel::pool::max_pool3d_with_indices_backward(
            x,
            output_grad,
            indices,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        ))
    }

    fn adaptive_avg_pool3d(x: FloatTensor<Self>, output_size: [usize; 3]) -> FloatTensor<Self> {
        kernel::pool::adaptive_avg_pool3d(x, output_size)
    }

    fn adaptive_avg_pool3d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        kernel::pool::adaptive_avg_pool3d_backward(x, grad)
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

    fn attention(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        mask: Option<BoolTensor<Self>>,
    ) -> FloatTensor<Self> {
        let out_dtype = query.dtype;
        kernel::attention::flash_attention(query, key, value, mask, out_dtype)
            .expect("Kernel to never fail")
    }
}
