use crate::{
    CubeBackend,
    kernel::{self, conv::ConvTranspose2dStrategy},
};
use burn_backend::tensor::{BoolTensor, FloatTensor, IntTensor};
use burn_backend::{
    Shape, TensorMetadata,
    ops::{
        AttentionModuleOptions, BoolTensorOps, CausalAlignment, ConvOptions, ConvTransposeOptions,
        DeformConv2dBackward, DeformConvOptions, FloatTensorOps, InterpolateOptions,
        MaxPool2dBackward, MaxPool2dWithIndices, ModuleOps, attention::AttentionShapes,
    },
};
use burn_std::IntDType;

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
        let shapes = AttentionShapes::new(&query.shape(), &key.shape(), &value.shape());
        let AttentionShapes {
            batch,
            q_heads,
            kv_heads,
            seq_q,
            seq_k,
            head_dim,
            val_dim,
        } = shapes;
        let groups = shapes.groups();

        // Fall back to naive attention for features the flash kernel doesn't support.
        // Grouped K/V heads are handled below.
        if kernel::attention::flash_unsupported_options(&options, attn_bias.is_some(), seq_q, seq_k)
            .is_some()
        {
            return burn_backend::ops::attention::attention_fallback::<Self>(
                query, key, value, mask, attn_bias, options,
            );
        }

        let flash = |query, key, value, mask, options| {
            kernel::attention::attention(query, key, value, mask, None, options, Default::default())
                .expect("Kernel to never fail")
        };

        if groups == 1 {
            return flash(query, key, value, mask, options);
        }

        // Grouped-query attention. A bottom-right causal mask hides nothing from a single
        // query row (decode), so it can be dropped there.
        let causal = options.is_causal
            && !(seq_q == 1 && options.causal_alignment == CausalAlignment::BottomRight);

        if !causal {
            // Fold the query heads sharing a K/V head into the row dimension, so the
            // kernel runs plain attention over `kv_heads` heads without repeating K/V.
            let fold =
                |rows: usize, cols: usize| Shape::new([batch, kv_heads, groups * rows, cols]);
            let query = Self::float_reshape(query, fold(seq_q, head_dim));
            let mask = mask.map(|mask| {
                let mask = Self::bool_expand(mask, Shape::new([batch, q_heads, seq_q, seq_k]));
                Self::bool_reshape(mask, fold(seq_q, seq_k))
            });
            let options = AttentionModuleOptions {
                is_causal: false,
                ..options
            };
            let out = flash(query, key, value, mask, options);
            return Self::float_reshape(out, Shape::new([batch, q_heads, seq_q, val_dim]));
        }

        // Causal over several query rows: folded rows would break the kernel's row
        // index. Until the kernel indexes K/V heads itself, repeat them.
        let repeat = |tensor: FloatTensor<Self>, dim: usize| {
            let tensor = Self::float_reshape(tensor, Shape::new([batch, kv_heads, 1, seq_k, dim]));
            let tensor =
                Self::float_expand(tensor, Shape::new([batch, kv_heads, groups, seq_k, dim]));
            Self::float_reshape(tensor, Shape::new([batch, q_heads, seq_k, dim]))
        };
        flash(
            query,
            repeat(key, head_dim),
            repeat(value, val_dim),
            mask,
            options,
        )
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
        kernel::ctc::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank)
    }

    fn ctc_loss_backward(
        log_probs: FloatTensor<Self>,
        targets: IntTensor<Self>,
        input_lengths: IntTensor<Self>,
        target_lengths: IntTensor<Self>,
        grad_loss: FloatTensor<Self>,
        blank: usize,
    ) -> FloatTensor<Self> {
        let (log_alpha_full, log_beta_full, nll) = kernel::ctc::ctc_alpha_beta(
            log_probs.clone(),
            targets.clone(),
            input_lengths.clone(),
            target_lengths,
            blank,
        );
        burn_backend::ops::ctc::ctc_grad_from_alpha_beta_default::<Self>(
            log_probs,
            targets,
            input_lengths,
            grad_loss,
            log_alpha_full,
            log_beta_full,
            nll,
            blank,
        )
    }
}
