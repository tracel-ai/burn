use crate::{LibTorch, TchTensor, element::TchElement};
use burn_backend::{
    TensorMetadata,
    ops::{
        ConvOptions, ConvTransposeOptions, DeformConv2dBackward, DeformConvOptions,
        InterpolateMode, InterpolateOptions, MaxPool1dWithIndices, MaxPool2dBackward,
        MaxPool2dWithIndices, ModuleOps,
    },
};

impl<E: TchElement> ModuleOps<Self> for LibTorch<E> {
    fn embedding(weights: TchTensor, indices: TchTensor) -> TchTensor {
        // Workaround for MPS "Placeholder storage has not been allocated" error.
        // See: https://github.com/pytorch/pytorch/issues/123995
        // MPS uses lazy allocation and the embedding operation (which uses index_select)
        // can fail if the tensors haven't been materialized yet.
        // We work around this by performing the embedding on CPU and transferring back to MPS.
        if matches!(weights.tensor.device(), tch::Device::Mps) {
            let cpu_weights = weights.tensor.to(tch::Device::Cpu);
            let cpu_indices = indices.tensor.to(tch::Device::Cpu);
            let result = tch::Tensor::embedding(&cpu_weights, &cpu_indices, -1, false, false)
                .to(tch::Device::Mps);
            return TchTensor::new(result);
        }

        let tensor = tch::Tensor::embedding(&weights.tensor, &indices.tensor, -1, false, false);
        TchTensor::new(tensor)
    }

    fn embedding_backward(weights: TchTensor, output: TchTensor, indices: TchTensor) -> TchTensor {
        let [n_embedding, _d_model] = weights.shape().dims();

        // Workaround for MPS "Placeholder storage has not been allocated" error.
        // See: https://github.com/pytorch/pytorch/issues/123995
        if matches!(output.tensor.device(), tch::Device::Mps) {
            let cpu_output = output.tensor.to(tch::Device::Cpu);
            let cpu_indices = indices.tensor.to(tch::Device::Cpu);
            let result = tch::Tensor::embedding_backward(
                &cpu_output,
                &cpu_indices,
                n_embedding as i64,
                -1,
                false,
                false,
            )
            .to(tch::Device::Mps);
            return TchTensor::new(result);
        }

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
        x: TchTensor,
        weight: TchTensor,
        bias: Option<TchTensor>,
        options: ConvOptions<1>,
    ) -> TchTensor {
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
        x: TchTensor,
        weight: TchTensor,
        bias: Option<TchTensor>,
        options: ConvOptions<2>,
    ) -> TchTensor {
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
        x: TchTensor,
        weight: TchTensor,
        bias: Option<TchTensor>,
        options: ConvOptions<3>,
    ) -> TchTensor {
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
        _x: TchTensor,
        _offset: TchTensor,
        _weight: TchTensor,
        _mask: Option<TchTensor>,
        _bias: Option<TchTensor>,
        _options: DeformConvOptions<2>,
    ) -> TchTensor {
        unimplemented!("Torch bindings don't support deform_conv2d");
    }

    fn deform_conv2d_backward(
        _x: TchTensor,
        _offset: TchTensor,
        _weight: TchTensor,
        _mask: Option<TchTensor>,
        _bias: Option<TchTensor>,
        _out_grad: TchTensor,
        _options: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<Self> {
        unimplemented!("Torch bindings don't support deform_conv2d");
    }

    fn conv_transpose1d(
        x: TchTensor,
        weight: TchTensor,
        bias: Option<TchTensor>,
        options: ConvTransposeOptions<1>,
    ) -> TchTensor {
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
        x: TchTensor,
        weight: TchTensor,
        bias: Option<TchTensor>,
        options: ConvTransposeOptions<2>,
    ) -> TchTensor {
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
        x: TchTensor,
        weight: TchTensor,
        bias: Option<TchTensor>,
        options: ConvTransposeOptions<3>,
    ) -> TchTensor {
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
        x: TchTensor,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> TchTensor {
        let tensor = tch::Tensor::avg_pool1d(
            &x.tensor,
            [kernel_size as i64],
            [stride as i64],
            [padding as i64],
            ceil_mode,
            count_include_pad,
        );

        TchTensor::new(tensor)
    }
    fn avg_pool2d(
        x: TchTensor,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> TchTensor {
        let tensor = tch::Tensor::avg_pool2d(
            &x.tensor,
            [kernel_size[0] as i64, kernel_size[1] as i64],
            [stride[0] as i64, stride[1] as i64],
            [padding[0] as i64, padding[1] as i64],
            ceil_mode,
            count_include_pad,
            None,
        );

        TchTensor::new(tensor)
    }

    fn avg_pool2d_backward(
        x: TchTensor,
        grad: TchTensor,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> TchTensor {
        let tensor = tch::Tensor::avg_pool2d_backward(
            &x.tensor,
            &grad.tensor,
            [kernel_size[0] as i64, kernel_size[1] as i64],
            [stride[0] as i64, stride[1] as i64],
            [padding[0] as i64, padding[1] as i64],
            ceil_mode,
            count_include_pad,
            None,
        );

        TchTensor::new(tensor)
    }

    fn max_pool1d(
        x: TchTensor,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        ceil_mode: bool,
    ) -> TchTensor {
        let tensor = tch::Tensor::max_pool1d(
            &x.tensor,
            kernel_size as i64,
            stride as i64,
            padding as i64,
            dilation as i64,
            ceil_mode,
        );

        TchTensor::new(tensor)
    }

    fn max_pool1d_with_indices(
        x: TchTensor,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        ceil_mode: bool,
    ) -> MaxPool1dWithIndices<Self> {
        let (tensor, indices) = tch::Tensor::max_pool1d_with_indices(
            &x.tensor,
            kernel_size as i64,
            stride as i64,
            padding as i64,
            dilation as i64,
            ceil_mode,
        );

        MaxPool1dWithIndices::new(TchTensor::new(tensor), TchTensor::new(indices))
    }

    fn max_pool2d(
        x: TchTensor,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> TchTensor {
        let tensor = tch::Tensor::max_pool2d(
            &x.tensor,
            [kernel_size[0] as i64, kernel_size[1] as i64],
            [stride[0] as i64, stride[1] as i64],
            [padding[0] as i64, padding[1] as i64],
            [dilation[0] as i64, dilation[1] as i64],
            ceil_mode,
        );

        TchTensor::new(tensor)
    }

    fn max_pool2d_with_indices(
        x: TchTensor,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> MaxPool2dWithIndices<Self> {
        let (tensor, indices) = tch::Tensor::max_pool2d_with_indices(
            &x.tensor,
            [kernel_size[0] as i64, kernel_size[1] as i64],
            [stride[0] as i64, stride[1] as i64],
            [padding[0] as i64, padding[1] as i64],
            [dilation[0] as i64, dilation[1] as i64],
            ceil_mode,
        );

        MaxPool2dWithIndices::new(TchTensor::new(tensor), TchTensor::new(indices))
    }

    fn max_pool2d_with_indices_backward(
        x: TchTensor,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
        output_grad: TchTensor,
        indices: TchTensor,
    ) -> MaxPool2dBackward<Self> {
        let grad = tch::Tensor::max_pool2d_with_indices_backward(
            &x.tensor,
            &output_grad.tensor,
            [kernel_size[0] as i64, kernel_size[1] as i64],
            [stride[0] as i64, stride[1] as i64],
            [padding[0] as i64, padding[1] as i64],
            [dilation[0] as i64, dilation[1] as i64],
            ceil_mode,
            &indices.tensor,
        );

        MaxPool2dBackward::new(TchTensor::new(grad))
    }

    fn adaptive_avg_pool2d(x: TchTensor, output_size: [usize; 2]) -> TchTensor {
        let tensor = tch::Tensor::adaptive_avg_pool2d(&x.tensor, output_size.map(|e| e as i64));

        TchTensor::new(tensor)
    }

    fn adaptive_avg_pool2d_backward(x: TchTensor, grad: TchTensor) -> TchTensor {
        let tensor = tch::Tensor::internal_adaptive_avg_pool2d_backward(&x.tensor, &grad.tensor);

        TchTensor::new(tensor)
    }

    fn adaptive_avg_pool1d(x: TchTensor, output_size: usize) -> TchTensor {
        let tensor = tch::Tensor::adaptive_avg_pool1d(&x.tensor, output_size as i64);

        TchTensor::new(tensor)
    }

    fn interpolate(
        x: TchTensor,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> TchTensor {
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
        x: TchTensor,
        grad: TchTensor,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> TchTensor {
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
