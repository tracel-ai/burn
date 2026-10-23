use crate::{
    CubeBackend,
    kernel::{
        self,
        conv::{ConvTranspose2dStrategy, ConvTranspose3dStrategy},
    },
};
use burn_backend::tensor::{BoolTensor, FloatTensor, IntTensor};
use burn_backend::{
    TensorMetadata,
    ops::{
        AttentionModuleOptions, ConvOptions, ConvTransposeOptions, DeformConv2dBackward,
        DeformConvOptions, FloatTensorOps, InterpolateOptions, MaxPool2dBackward,
        MaxPool2dWithIndices, MaxPool3dBackward, MaxPool3dWithIndices, ModuleOps,
    },
};
use burn_std::{FloatDType, IntDType};

impl ModuleOps<Self> for CubeBackend {
    fn conv1d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<1>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv_forward::<1>(x, weight, bias, options, Default::default()).unwrap()
    }

    fn conv1d_x_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: ConvOptions<1>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv_data_backward(
            output_grad,
            weight,
            x.shape(),
            options,
            Default::default(),
        )
        .unwrap()
    }

    fn conv1d_weight_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: ConvOptions<1>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv_weight_backward::<1>(
            x,
            output_grad,
            weight.shape(),
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
        kernel::conv::conv_forward::<2>(x, weight, bias, options, Default::default()).unwrap()
    }

    fn conv2d_x_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: ConvOptions<2>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv_data_backward(
            output_grad,
            weight,
            x.shape(),
            options,
            Default::default(),
        )
        .unwrap()
    }

    fn conv2d_weight_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: ConvOptions<2>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv_weight_backward::<2>(
            x,
            output_grad,
            weight.shape(),
            options,
            Default::default(),
        )
        .unwrap()
    }

    /// Sums gradients `[batch, output_channels, height, width]` over batch and
    /// spatial axes, returning `[output_channels]` with the same dtype and device.
    /// `bias` has shape `[output_channels]`. The unused input `x` has layout
    /// `[batch, input_channels, input_height, input_width]`.
    fn conv2d_bias_backward(
        _x: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let grad = Self::float_sum_dims(output_grad, &[0, 2, 3]);
        Self::float_reshape(grad, bias.shape())
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
        kernel::conv::conv_forward::<3>(x, weight, bias, options, Default::default()).unwrap()
    }

    fn conv3d_x_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: ConvOptions<3>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv_data_backward(
            output_grad,
            weight,
            x.shape(),
            options,
            Default::default(),
        )
        .unwrap()
    }

    fn conv3d_weight_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        options: ConvOptions<3>,
    ) -> FloatTensor<Self> {
        kernel::conv::conv_weight_backward::<3>(
            x,
            output_grad,
            weight.shape(),
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
        kernel::conv::conv_transpose3d(x, weight, bias, options, ConvTranspose3dStrategy::default())
            .unwrap()
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
        indices_dtype: IntDType,
    ) -> MaxPool2dWithIndices<Self> {
        let (output, indices) = kernel::pool::max_pool2d_with_indices(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            indices_dtype.into(),
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

    fn adaptive_avg_pool3d(x: FloatTensor<Self>, output_size: [usize; 3]) -> FloatTensor<Self> {
        kernel::pool::adaptive_avg_pool3d(x, output_size)
    }

    fn adaptive_avg_pool3d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        kernel::pool::adaptive_avg_pool3d_backward(x, grad)
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
        let (output, _indices) = kernel::pool::max_pool3d_with_indices(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            burn_backend::DType::I32,
        );
        output
    }

    fn max_pool3d_with_indices(
        x: FloatTensor<Self>,
        kernel_size: [usize; 3],
        stride: [usize; 3],
        padding: [usize; 3],
        dilation: [usize; 3],
        ceil_mode: bool,
        indices_dtype: IntDType,
    ) -> MaxPool3dWithIndices<Self> {
        let (output, indices) = kernel::pool::max_pool3d_with_indices(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            indices_dtype.into(),
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

    fn interpolate(
        x: FloatTensor<Self>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Self> {
        kernel::interpolate::interpolate(x, output_size, options, Default::default()).unwrap()
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
        attn_bias: Option<FloatTensor<Self>>,
        options: AttentionModuleOptions,
    ) -> FloatTensor<Self> {
        // Fall back to naive attention for features the flash kernel doesn't support.
        if attn_bias.is_some() || options.softcap.is_some() || options.scale.is_some() {
            return burn_backend::ops::attention::attention_fallback::<Self>(
                query, key, value, mask, attn_bias, options,
            );
        }

        kernel::attention::attention(
            query,
            key,
            value,
            mask,
            attn_bias,
            options,
            Default::default(),
        )
        .expect("Kernel to never fail")
    }

    fn has_ctc_loss_backward() -> bool {
        true
    }

    fn ctc_loss(
        log_probs: FloatTensor<Self>,
        targets: IntTensor<Self>,
        input_lengths: IntTensor<Self>,
        target_lengths: IntTensor<Self>,
        blank: usize,
    ) -> FloatTensor<Self> {
        let dtype = log_probs.dtype.into();
        let log_probs = match dtype {
            FloatDType::F16 | FloatDType::BF16 => Self::float_cast(log_probs, FloatDType::F32),
            _ => log_probs,
        };
        let loss = kernel::ctc::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank);

        Self::float_cast(loss, dtype)
    }

    fn ctc_loss_backward(
        log_probs: FloatTensor<Self>,
        targets: IntTensor<Self>,
        input_lengths: IntTensor<Self>,
        target_lengths: IntTensor<Self>,
        grad_loss: FloatTensor<Self>,
        blank: usize,
    ) -> FloatTensor<Self> {
        let dtype = log_probs.dtype.into();
        let log_probs = match dtype {
            FloatDType::F16 | FloatDType::BF16 => Self::float_cast(log_probs, FloatDType::F32),
            _ => log_probs,
        };
        let grad_loss = Self::float_cast(grad_loss, log_probs.dtype.into());
        let (log_alpha_full, log_beta_full, nll) = kernel::ctc::ctc_alpha_beta(
            log_probs.clone(),
            targets.clone(),
            input_lengths.clone(),
            target_lengths,
            blank,
        );
        let grad = burn_backend::ops::ctc::ctc_grad_from_alpha_beta_default::<Self>(
            log_probs,
            targets,
            input_lengths,
            grad_loss,
            log_alpha_full,
            log_beta_full,
            nll,
            blank,
        );

        Self::float_cast(grad, dtype)
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::{
        CubeDevice,
        ops::{from_data, into_data_sync, permute_nhwc_to_nchw},
    };
    use burn_backend::{DType, TensorData};

    #[test]
    fn conv2d_bias_gradient_preserves_channels_across_layouts() {
        let device = CubeDevice::cuda(0).unwrap();
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for channels_last in [false, true] {
                let values = (0..2 * 3 * 2 * 4)
                    .map(|i| ((i % 11) as f32 - 5.0) / 8.0)
                    .collect::<Vec<_>>();
                let mut expected = [0.0_f32; 3];
                for b in 0..2 {
                    for (c, channel_sum) in expected.iter_mut().enumerate() {
                        for y in 0..2 {
                            for x in 0..4 {
                                *channel_sum += values[((b * 3 + c) * 2 + y) * 4 + x];
                            }
                        }
                    }
                }
                let grad = if channels_last {
                    let mut stored = Vec::new();
                    for b in 0..2 {
                        for y in 0..2 {
                            for x in 0..4 {
                                for c in 0..3 {
                                    stored.push(values[((b * 3 + c) * 2 + y) * 4 + x]);
                                }
                            }
                        }
                    }
                    permute_nhwc_to_nchw(from_data(
                        TensorData::new(stored, [2, 2, 4, 3]).convert_dtype(dtype),
                        &device,
                    ))
                } else {
                    from_data(
                        TensorData::new(values, [2, 3, 2, 4]).convert_dtype(dtype),
                        &device,
                    )
                };
                let input = from_data(
                    TensorData::new(vec![0.0_f32; 16], [2, 1, 2, 4]).convert_dtype(dtype),
                    &device,
                );
                let bias = from_data(
                    TensorData::new(vec![0.0_f32; 3], [3]).convert_dtype(dtype),
                    &device,
                );
                let actual = CubeBackend::conv2d_bias_backward(input, bias, grad);
                assert_eq!(actual.shape(), [3].into());
                assert_eq!(actual.dtype, dtype);
                let actual = into_data_sync(actual)
                    .convert::<f32>()
                    .try_to_vec::<f32>()
                    .unwrap();
                assert_eq!(actual, expected);
            }
        }
    }
}
