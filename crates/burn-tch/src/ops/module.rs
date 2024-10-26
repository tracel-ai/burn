use crate::{element::TchElement, LibTorch, QuantElement, TchTensor};
use burn_tensor::ops::{
    ConvOptions, ConvTransposeOptions, DeformConv2dBackward, DeformConvOptions, InterpolateMode,
    InterpolateOptions, MaxPool1dWithIndices, MaxPool2dBackward, MaxPool2dWithIndices, ModuleOps,
};

impl<E: TchElement, Q: QuantElement> ModuleOps<Self> for LibTorch<E, Q> {
    fn embedding(weights: TchTensor<E>, indices: TchTensor<i64>) -> TchTensor<E> {
        let tensor = tch::Tensor::embedding(&weights.tensor, &indices.tensor, -1, false, false);

        TchTensor::new(tensor)
    }

    fn embedding_backward(
        weights: TchTensor<E>,
        output: TchTensor<E>,
        indices: TchTensor<i64>,
    ) -> TchTensor<E> {
        let [n_embedding, _d_model] = weights.shape().dims();
        let tensor = tch::Tensor::embedding_backward(
            &output.tensor,
            &indices.tensor,
            n_embedding as i64,
            -1,
            false,
            false,
        );

        TchTensor::new(tensor)
    }

    fn conv1d(
        x: TchTensor<E>,
        weight: TchTensor<E>,
        bias: Option<TchTensor<E>>,
        options: ConvOptions<1>,
    ) -> TchTensor<E> {
        let tensor = tch::Tensor::conv1d(
            &x.tensor,
            &weight.tensor,
            bias.map(|t| t.tensor),
            options.stride.map(|i| i as i64),
            options.padding.map(|i| i as i64),
            options.dilation.map(|i| i as i64),
            options.groups as i64,
        );

        TchTensor::new(tensor)
    }

    fn conv2d(
        x: TchTensor<E>,
        weight: TchTensor<E>,
        bias: Option<TchTensor<E>>,
        options: ConvOptions<2>,
    ) -> TchTensor<E> {
        let tensor = tch::Tensor::conv2d(
            &x.tensor,
            &weight.tensor,
            bias.map(|t| t.tensor),
            options.stride.map(|i| i as i64),
            options.padding.map(|i| i as i64),
            options.dilation.map(|i| i as i64),
            options.groups as i64,
        );

        TchTensor::new(tensor)
    }

    fn conv3d(
        x: TchTensor<E>,
        weight: TchTensor<E>,
        bias: Option<TchTensor<E>>,
        options: ConvOptions<3>,
    ) -> TchTensor<E> {
        let tensor = tch::Tensor::conv3d(
            &x.tensor,
            &weight.tensor,
            bias.map(|t| t.tensor),
            options.stride.map(|i| i as i64),
            options.padding.map(|i| i as i64),
            options.dilation.map(|i| i as i64),
            options.groups as i64,
        );

        TchTensor::new(tensor)
    }

    fn deform_conv2d(
        _x: TchTensor<E>,
        _offset: TchTensor<E>,
        _weight: TchTensor<E>,
        _mask: Option<TchTensor<E>>,
        _bias: Option<TchTensor<E>>,
        _options: DeformConvOptions<2>,
    ) -> TchTensor<E> {
        unimplemented!("Torch bindings don't support deform_conv2d");
    }

    fn deform_conv2d_backward(
        _x: TchTensor<E>,
        _offset: TchTensor<E>,
        _weight: TchTensor<E>,
        _mask: Option<TchTensor<E>>,
        _bias: Option<TchTensor<E>>,
        _out_grad: TchTensor<E>,
        _options: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<Self> {
        unimplemented!("Torch bindings don't support deform_conv2d");
    }

    fn conv_transpose1d(
        x: TchTensor<E>,
        weight: TchTensor<E>,
        bias: Option<TchTensor<E>>,
        options: ConvTransposeOptions<1>,
    ) -> TchTensor<E> {
        let tensor = tch::Tensor::conv_transpose1d(
            &x.tensor,
            &weight.tensor,
            bias.map(|t| t.tensor),
            options.stride.map(|i| i as i64),
            options.padding.map(|i| i as i64),
            options.padding_out.map(|i| i as i64),
            options.groups as i64,
            options.dilation.map(|i| i as i64),
        );

        TchTensor::new(tensor)
    }

    fn conv_transpose2d(
        x: TchTensor<E>,
        weight: TchTensor<E>,
        bias: Option<TchTensor<E>>,
        options: ConvTransposeOptions<2>,
    ) -> TchTensor<E> {
        let tensor = tch::Tensor::conv_transpose2d(
            &x.tensor,
            &weight.tensor,
            bias.map(|t| t.tensor),
            options.stride.map(|i| i as i64),
            options.padding.map(|i| i as i64),
            options.padding_out.map(|i| i as i64),
            options.groups as i64,
            options.dilation.map(|i| i as i64),
        );

        TchTensor::new(tensor)
    }

    fn conv_transpose3d(
        x: TchTensor<E>,
        weight: TchTensor<E>,
        bias: Option<TchTensor<E>>,
        options: ConvTransposeOptions<3>,
    ) -> TchTensor<E> {
        let tensor = tch::Tensor::conv_transpose3d(
            &x.tensor,
            &weight.tensor,
            bias.map(|t| t.tensor),
            options.stride.map(|i| i as i64),
            options.padding.map(|i| i as i64),
            options.padding_out.map(|i| i as i64),
            options.groups as i64,
            options.dilation.map(|i| i as i64),
        );

        TchTensor::new(tensor)
    }

    fn avg_pool1d(
        x: TchTensor<E>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
    ) -> TchTensor<E> {
        let tensor = tch::Tensor::avg_pool1d(
            &x.tensor,
            [kernel_size as i64],
            [stride as i64],
            [padding as i64],
            false,
            count_include_pad,
        );

        TchTensor::new(tensor)
    }
    fn avg_pool2d(
        x: TchTensor<E>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> TchTensor<E> {
        let tensor = tch::Tensor::avg_pool2d(
            &x.tensor,
            [kernel_size[0] as i64, kernel_size[1] as i64],
            [stride[0] as i64, stride[1] as i64],
            [padding[0] as i64, padding[1] as i64],
            false,
            count_include_pad,
            None,
        );

        TchTensor::new(tensor)
    }

    fn avg_pool2d_backward(
        x: TchTensor<E>,
        grad: TchTensor<E>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> TchTensor<E> {
        let tensor = tch::Tensor::avg_pool2d_backward(
            &x.tensor,
            &grad.tensor,
            [kernel_size[0] as i64, kernel_size[1] as i64],
            [stride[0] as i64, stride[1] as i64],
            [padding[0] as i64, padding[1] as i64],
            false,
            count_include_pad,
            None,
        );

        TchTensor::new(tensor)
    }

    fn max_pool1d(
        x: TchTensor<E>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> TchTensor<E> {
        let tensor = tch::Tensor::max_pool1d(
            &x.tensor,
            kernel_size as i64,
            stride as i64,
            padding as i64,
            dilation as i64,
            false,
        );

        TchTensor::new(tensor)
    }

    fn max_pool1d_with_indices(
        x: TchTensor<E>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> MaxPool1dWithIndices<LibTorch<E, Q>> {
        let (tensor, indices) = tch::Tensor::max_pool1d_with_indices(
            &x.tensor,
            kernel_size as i64,
            stride as i64,
            padding as i64,
            dilation as i64,
            false,
        );

        MaxPool1dWithIndices::new(TchTensor::new(tensor), TchTensor::new(indices))
    }

    fn max_pool2d(
        x: TchTensor<E>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> TchTensor<E> {
        let tensor = tch::Tensor::max_pool2d(
            &x.tensor,
            [kernel_size[0] as i64, kernel_size[1] as i64],
            [stride[0] as i64, stride[1] as i64],
            [padding[0] as i64, padding[1] as i64],
            [dilation[0] as i64, dilation[1] as i64],
            false,
        );

        TchTensor::new(tensor)
    }

    fn max_pool2d_with_indices(
        x: TchTensor<E>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> MaxPool2dWithIndices<LibTorch<E, Q>> {
        let (tensor, indices) = tch::Tensor::max_pool2d_with_indices(
            &x.tensor,
            [kernel_size[0] as i64, kernel_size[1] as i64],
            [stride[0] as i64, stride[1] as i64],
            [padding[0] as i64, padding[1] as i64],
            [dilation[0] as i64, dilation[1] as i64],
            false,
        );

        MaxPool2dWithIndices::new(TchTensor::new(tensor), TchTensor::new(indices))
    }

    fn max_pool2d_with_indices_backward(
        x: TchTensor<E>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        output_grad: TchTensor<E>,
        indices: TchTensor<i64>,
    ) -> MaxPool2dBackward<LibTorch<E, Q>> {
        let grad = tch::Tensor::max_pool2d_with_indices_backward(
            &x.tensor,
            &output_grad.tensor,
            [kernel_size[0] as i64, kernel_size[1] as i64],
            [stride[0] as i64, stride[1] as i64],
            [padding[0] as i64, padding[1] as i64],
            [dilation[0] as i64, dilation[1] as i64],
            false,
            &indices.tensor,
        );

        MaxPool2dBackward::new(TchTensor::new(grad))
    }

    fn adaptive_avg_pool2d(x: TchTensor<E>, output_size: [usize; 2]) -> TchTensor<E> {
        let tensor = tch::Tensor::adaptive_avg_pool2d(&x.tensor, output_size.map(|e| e as i64));

        TchTensor::new(tensor)
    }

    fn adaptive_avg_pool2d_backward(x: TchTensor<E>, grad: TchTensor<E>) -> TchTensor<E> {
        let tensor = tch::Tensor::internal_adaptive_avg_pool2d_backward(&x.tensor, &grad.tensor);

        TchTensor::new(tensor)
    }

    fn adaptive_avg_pool1d(x: TchTensor<E>, output_size: usize) -> TchTensor<E> {
        let tensor = tch::Tensor::adaptive_avg_pool1d(&x.tensor, output_size as i64);

        TchTensor::new(tensor)
    }

    fn interpolate(
        x: TchTensor<E>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> TchTensor<E> {
        let output_size = output_size.map(|e| e as i64);

        let tensor = match options.mode {
            InterpolateMode::Nearest => {
                tch::Tensor::upsample_nearest2d(&x.tensor, output_size, None, None)
            }
            InterpolateMode::Bilinear => {
                tch::Tensor::upsample_bilinear2d(&x.tensor, output_size, true, None, None)
            }
            InterpolateMode::Bicubic => {
                tch::Tensor::upsample_bicubic2d(&x.tensor, output_size, true, None, None)
            }
        };

        TchTensor::new(tensor)
    }

    fn interpolate_backward(
        x: TchTensor<E>,
        grad: TchTensor<E>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> TchTensor<E> {
        let output_size = output_size.map(|e| e as i64);
        let [n, c, h_in, w_in] = x.shape().dims();
        let input_size = [n as i64, c as i64, h_in as i64, w_in as i64];

        let tensor = match options.mode {
            InterpolateMode::Nearest => tch::Tensor::upsample_nearest2d_backward(
                &grad.tensor,
                output_size,
                input_size,
                None,
                None,
            ),
            InterpolateMode::Bilinear => tch::Tensor::upsample_bilinear2d_backward(
                &grad.tensor,
                output_size,
                input_size,
                true,
                None,
                None,
            ),
            InterpolateMode::Bicubic => tch::Tensor::upsample_bicubic2d_backward(
                &grad.tensor,
                output_size,
                input_size,
                true,
                None,
                None,
            ),
        };

        TchTensor::new(tensor)
    }
}
