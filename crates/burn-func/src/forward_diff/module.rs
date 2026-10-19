use burn_backend::{
    Backend, BoolStore, Distribution, ExecutionError, FloatDType, IntDType, Scalar, Shape, Slice,
    TensorData, TensorMetadata,
    distributed::DistributedOps,
    ops::{
        ActivationOps, AttentionModuleOptions, BoolTensorOps, ConvOptions, ConvTransposeOptions,
        DeformConv2dBackward, DeformConvOptions, IntTensorOps, InterpolateOptions,
        MaxPool2dBackward, MaxPool2dWithIndices, MaxPool3dBackward, MaxPool3dWithIndices,
        ModuleOps, QTensorOps, TransactionOps,
    },
    quantization::QuantizationParametersPrimitive,
    tensor::{BoolTensor, FloatTensor, IntTensor},
};
use burn_std::quantization::QuantScheme;
use core::future::Future;

use super::{DualFloatTensor, ForwardDiff};

type Dual<B> = DualFloatTensor<B>;

fn add_or_set<B: Backend>(
    current: Option<FloatTensor<B>>,
    value: FloatTensor<B>,
) -> Option<FloatTensor<B>> {
    Some(match current {
        Some(current) => B::float_add(current, value),
        None => value,
    })
}

fn max_pool2d_tangent<B: Backend>(
    dx: FloatTensor<B>,
    indices: IntTensor<B>,
    out_shape: &Shape,
) -> FloatTensor<B> {
    let in_dims = dx.shape();
    let [n, c, h_in, w_in] = [in_dims[0], in_dims[1], in_dims[2], in_dims[3]];
    let [_, _, h_out, w_out] = [out_shape[0], out_shape[1], out_shape[2], out_shape[3]];
    let dx_flat = B::float_reshape(dx, Shape::from(alloc::vec![n, c, h_in * w_in]));
    let indices_flat = B::int_reshape(indices, Shape::from(alloc::vec![n, c, h_out * w_out]));
    let gathered = B::float_gather(2, dx_flat, indices_flat);
    B::float_reshape(gathered, Shape::from(alloc::vec![n, c, h_out, w_out]))
}

fn max_pool3d_tangent<B: Backend>(
    dx: FloatTensor<B>,
    indices: IntTensor<B>,
    out_shape: &Shape,
) -> FloatTensor<B> {
    let in_dims = dx.shape();
    let [n, c, d_in, h_in, w_in] = [in_dims[0], in_dims[1], in_dims[2], in_dims[3], in_dims[4]];
    let [_, _, d_out, h_out, w_out] = [
        out_shape[0],
        out_shape[1],
        out_shape[2],
        out_shape[3],
        out_shape[4],
    ];
    let dx_flat = B::float_reshape(dx, Shape::from(alloc::vec![n, c, d_in * h_in * w_in]));
    let indices_flat = B::int_reshape(
        indices,
        Shape::from(alloc::vec![n, c, d_out * h_out * w_out]),
    );
    let gathered = B::float_gather(2, dx_flat, indices_flat);
    B::float_reshape(
        gathered,
        Shape::from(alloc::vec![n, c, d_out, h_out, w_out]),
    )
}

impl<B: Backend> ActivationOps<ForwardDiff<B>> for ForwardDiff<B> {}

impl<B: Backend> ModuleOps<ForwardDiff<B>> for ForwardDiff<B> {
    fn conv2d(
        x: Dual<B>,
        weight: Dual<B>,
        bias: Option<Dual<B>>,
        options: ConvOptions<2>,
    ) -> Dual<B> {
        let primal = B::conv2d(
            x.primal.clone(),
            weight.primal.clone(),
            bias.as_ref().map(|bias| bias.primal.clone()),
            options.clone(),
        );
        let mut tangent = None;
        if let Some(dx) = x.tangent {
            tangent = Some(B::conv2d(dx, weight.primal.clone(), None, options.clone()));
        }
        if let Some(dw) = weight.tangent {
            let term = B::conv2d(x.primal.clone(), dw, None, options.clone());
            tangent = add_or_set::<B>(tangent, term);
        }
        if let Some(db) = bias.and_then(|bias| bias.tangent) {
            let shape = primal.shape();
            let channels = shape[1];
            let db = B::float_reshape(db, Shape::new([1, channels, 1, 1]));
            let db = B::float_expand(db, shape);
            tangent = add_or_set::<B>(tangent, db);
        }
        DualFloatTensor { primal, tangent }
    }

    fn deform_conv2d(
        x: Dual<B>,
        offset: Dual<B>,
        weight: Dual<B>,
        mask: Option<Dual<B>>,
        bias: Option<Dual<B>>,
        options: DeformConvOptions<2>,
    ) -> Dual<B> {
        DualFloatTensor::constant(B::deform_conv2d(
            x.primal,
            offset.primal,
            weight.primal,
            mask.map(|mask| mask.primal),
            bias.map(|bias| bias.primal),
            options,
        ))
    }

    fn deform_conv2d_backward(
        x: Dual<B>,
        offset: Dual<B>,
        weight: Dual<B>,
        mask: Option<Dual<B>>,
        bias: Option<Dual<B>>,
        output_grad: Dual<B>,
        options: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<ForwardDiff<B>> {
        let grads = B::deform_conv2d_backward(
            x.primal,
            offset.primal,
            weight.primal,
            mask.map(|mask| mask.primal),
            bias.map(|bias| bias.primal),
            output_grad.primal,
            options,
        );
        DeformConv2dBackward::new(
            DualFloatTensor::constant(grads.x_grad),
            DualFloatTensor::constant(grads.offset_grad),
            DualFloatTensor::constant(grads.weight_grad),
            grads.mask_grad.map(DualFloatTensor::constant),
            grads.bias_grad.map(DualFloatTensor::constant),
        )
    }

    fn conv3d(
        x: Dual<B>,
        weight: Dual<B>,
        bias: Option<Dual<B>>,
        options: ConvOptions<3>,
    ) -> Dual<B> {
        let primal = B::conv3d(
            x.primal.clone(),
            weight.primal.clone(),
            bias.as_ref().map(|bias| bias.primal.clone()),
            options.clone(),
        );
        let mut tangent = None;
        if let Some(dx) = x.tangent {
            tangent = Some(B::conv3d(dx, weight.primal.clone(), None, options.clone()));
        }
        if let Some(dw) = weight.tangent {
            let term = B::conv3d(x.primal.clone(), dw, None, options.clone());
            tangent = add_or_set::<B>(tangent, term);
        }
        if let Some(db) = bias.and_then(|bias| bias.tangent) {
            let shape = primal.shape();
            let channels = shape[1];
            let db = B::float_reshape(db, Shape::new([1, channels, 1, 1, 1]));
            let db = B::float_expand(db, shape);
            tangent = add_or_set::<B>(tangent, db);
        }
        DualFloatTensor { primal, tangent }
    }

    fn conv_transpose2d(
        x: Dual<B>,
        weight: Dual<B>,
        bias: Option<Dual<B>>,
        options: ConvTransposeOptions<2>,
    ) -> Dual<B> {
        let primal = B::conv_transpose2d(
            x.primal.clone(),
            weight.primal.clone(),
            bias.as_ref().map(|bias| bias.primal.clone()),
            options.clone(),
        );
        let mut tangent = None;
        if let Some(dx) = x.tangent {
            tangent = Some(B::conv_transpose2d(
                dx,
                weight.primal.clone(),
                None,
                options.clone(),
            ));
        }
        if let Some(dw) = weight.tangent {
            let term = B::conv_transpose2d(x.primal.clone(), dw, None, options.clone());
            tangent = add_or_set::<B>(tangent, term);
        }
        if let Some(db) = bias.and_then(|bias| bias.tangent) {
            let shape = primal.shape();
            let channels = shape[1];
            let db = B::float_reshape(db, Shape::new([1, channels, 1, 1]));
            let db = B::float_expand(db, shape);
            tangent = add_or_set::<B>(tangent, db);
        }
        DualFloatTensor { primal, tangent }
    }

    fn conv_transpose3d(
        x: Dual<B>,
        weight: Dual<B>,
        bias: Option<Dual<B>>,
        options: ConvTransposeOptions<3>,
    ) -> Dual<B> {
        let primal = B::conv_transpose3d(
            x.primal.clone(),
            weight.primal.clone(),
            bias.as_ref().map(|bias| bias.primal.clone()),
            options.clone(),
        );
        let mut tangent = None;
        if let Some(dx) = x.tangent {
            tangent = Some(B::conv_transpose3d(
                dx,
                weight.primal.clone(),
                None,
                options.clone(),
            ));
        }
        if let Some(dw) = weight.tangent {
            let term = B::conv_transpose3d(x.primal.clone(), dw, None, options.clone());
            tangent = add_or_set::<B>(tangent, term);
        }
        if let Some(db) = bias.and_then(|bias| bias.tangent) {
            let shape = primal.shape();
            let channels = shape[1];
            let db = B::float_reshape(db, Shape::new([1, channels, 1, 1, 1]));
            let db = B::float_expand(db, shape);
            tangent = add_or_set::<B>(tangent, db);
        }
        DualFloatTensor { primal, tangent }
    }

    fn avg_pool2d(
        x: Dual<B>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> Dual<B> {
        let primal = B::avg_pool2d(
            x.primal,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        );
        let tangent = x.tangent.map(|tangent| {
            B::avg_pool2d(
                tangent,
                kernel_size,
                stride,
                padding,
                count_include_pad,
                ceil_mode,
            )
        });
        DualFloatTensor { primal, tangent }
    }

    fn avg_pool2d_backward(
        x: Dual<B>,
        grad: Dual<B>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> Dual<B> {
        DualFloatTensor::constant(B::avg_pool2d_backward(
            x.primal,
            grad.primal,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        ))
    }

    fn avg_pool3d(
        x: Dual<B>,
        kernel_size: [usize; 3],
        stride: [usize; 3],
        padding: [usize; 3],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> Dual<B> {
        let primal = B::avg_pool3d(
            x.primal,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        );
        let tangent = x.tangent.map(|tangent| {
            B::avg_pool3d(
                tangent,
                kernel_size,
                stride,
                padding,
                count_include_pad,
                ceil_mode,
            )
        });
        DualFloatTensor { primal, tangent }
    }

    fn avg_pool3d_backward(
        x: Dual<B>,
        grad: Dual<B>,
        kernel_size: [usize; 3],
        stride: [usize; 3],
        padding: [usize; 3],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> Dual<B> {
        DualFloatTensor::constant(B::avg_pool3d_backward(
            x.primal,
            grad.primal,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        ))
    }

    fn adaptive_avg_pool2d(x: Dual<B>, output_size: [usize; 2]) -> Dual<B> {
        let primal = B::adaptive_avg_pool2d(x.primal, output_size);
        let tangent = x
            .tangent
            .map(|tangent| B::adaptive_avg_pool2d(tangent, output_size));
        DualFloatTensor { primal, tangent }
    }

    fn adaptive_avg_pool2d_backward(x: Dual<B>, grad: Dual<B>) -> Dual<B> {
        DualFloatTensor::constant(B::adaptive_avg_pool2d_backward(x.primal, grad.primal))
    }

    fn adaptive_avg_pool3d(x: Dual<B>, output_size: [usize; 3]) -> Dual<B> {
        let primal = B::adaptive_avg_pool3d(x.primal, output_size);
        let tangent = x
            .tangent
            .map(|tangent| B::adaptive_avg_pool3d(tangent, output_size));
        DualFloatTensor { primal, tangent }
    }

    fn adaptive_avg_pool3d_backward(x: Dual<B>, grad: Dual<B>) -> Dual<B> {
        DualFloatTensor::constant(B::adaptive_avg_pool3d_backward(x.primal, grad.primal))
    }

    fn max_pool2d(
        x: Dual<B>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> Dual<B> {
        match x.tangent {
            Some(dx) => {
                let settings = burn_backend::get_or_init_device_settings::<B>(&x.primal.device());
                let result = B::max_pool2d_with_indices(
                    x.primal,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    ceil_mode,
                    settings.int_dtype,
                );
                let tangent = max_pool2d_tangent::<B>(dx, result.indices, &result.output.shape());
                DualFloatTensor {
                    primal: result.output,
                    tangent: Some(tangent),
                }
            }
            None => DualFloatTensor::constant(B::max_pool2d(
                x.primal,
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
            )),
        }
    }

    fn max_pool2d_with_indices(
        x: Dual<B>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
        indices_dtype: IntDType,
    ) -> MaxPool2dWithIndices<ForwardDiff<B>> {
        let result = B::max_pool2d_with_indices(
            x.primal,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            indices_dtype,
        );
        let tangent = x
            .tangent
            .map(|dx| max_pool2d_tangent::<B>(dx, result.indices.clone(), &result.output.shape()));
        MaxPool2dWithIndices::new(
            DualFloatTensor {
                primal: result.output,
                tangent,
            },
            result.indices,
        )
    }

    fn max_pool2d_with_indices_backward(
        x: Dual<B>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
        output_grad: Dual<B>,
        indices: IntTensor<B>,
    ) -> MaxPool2dBackward<ForwardDiff<B>> {
        let result = B::max_pool2d_with_indices_backward(
            x.primal,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            output_grad.primal,
            indices,
        );
        MaxPool2dBackward::new(DualFloatTensor::constant(result.x_grad))
    }

    fn max_pool3d(
        x: Dual<B>,
        kernel_size: [usize; 3],
        stride: [usize; 3],
        padding: [usize; 3],
        dilation: [usize; 3],
        ceil_mode: bool,
    ) -> Dual<B> {
        match x.tangent {
            Some(dx) => {
                let settings = burn_backend::get_or_init_device_settings::<B>(&x.primal.device());
                let result = B::max_pool3d_with_indices(
                    x.primal,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    ceil_mode,
                    settings.int_dtype,
                );
                let tangent = max_pool3d_tangent::<B>(dx, result.indices, &result.output.shape());
                DualFloatTensor {
                    primal: result.output,
                    tangent: Some(tangent),
                }
            }
            None => DualFloatTensor::constant(B::max_pool3d(
                x.primal,
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
            )),
        }
    }

    fn max_pool3d_with_indices(
        x: Dual<B>,
        kernel_size: [usize; 3],
        stride: [usize; 3],
        padding: [usize; 3],
        dilation: [usize; 3],
        ceil_mode: bool,
        indices_dtype: IntDType,
    ) -> MaxPool3dWithIndices<ForwardDiff<B>> {
        let result = B::max_pool3d_with_indices(
            x.primal,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            indices_dtype,
        );
        let tangent = x
            .tangent
            .map(|dx| max_pool3d_tangent::<B>(dx, result.indices.clone(), &result.output.shape()));
        MaxPool3dWithIndices::new(
            DualFloatTensor {
                primal: result.output,
                tangent,
            },
            result.indices,
        )
    }

    fn max_pool3d_with_indices_backward(
        x: Dual<B>,
        kernel_size: [usize; 3],
        stride: [usize; 3],
        padding: [usize; 3],
        dilation: [usize; 3],
        ceil_mode: bool,
        output_grad: Dual<B>,
        indices: IntTensor<B>,
    ) -> MaxPool3dBackward<ForwardDiff<B>> {
        let result = B::max_pool3d_with_indices_backward(
            x.primal,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            output_grad.primal,
            indices,
        );
        MaxPool3dBackward::new(DualFloatTensor::constant(result.x_grad))
    }

    fn interpolate(x: Dual<B>, output_size: [usize; 2], options: InterpolateOptions) -> Dual<B> {
        let primal = B::interpolate(x.primal, output_size, options.clone());
        let tangent = x
            .tangent
            .map(|tangent| B::interpolate(tangent, output_size, options));
        DualFloatTensor { primal, tangent }
    }

    fn interpolate_backward(
        x: Dual<B>,
        grad: Dual<B>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> Dual<B> {
        DualFloatTensor::constant(B::interpolate_backward(
            x.primal,
            grad.primal,
            output_size,
            options,
        ))
    }

    fn attention(
        query: Dual<B>,
        key: Dual<B>,
        value: Dual<B>,
        mask: Option<BoolTensor<B>>,
        attn_bias: Option<Dual<B>>,
        options: AttentionModuleOptions,
    ) -> Dual<B> {
        burn_backend::ops::attention::attention_fallback::<ForwardDiff<B>>(
            query, key, value, mask, attn_bias, options,
        )
    }
}

impl<B: Backend> IntTensorOps<ForwardDiff<B>> for ForwardDiff<B> {
    fn int_empty(shape: Shape, device: &B::Device, dtype: IntDType) -> IntTensor<B> {
        B::int_empty(shape, device, dtype)
    }

    fn int_into_data(
        tensor: IntTensor<B>,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send {
        B::int_into_data(tensor)
    }

    fn int_from_data(data: TensorData, device: &B::Device) -> IntTensor<B> {
        B::int_from_data(data, device)
    }

    fn int_to_device(tensor: IntTensor<B>, device: &B::Device) -> IntTensor<B> {
        B::int_to_device(tensor, device)
    }

    fn int_reshape(tensor: IntTensor<B>, shape: Shape) -> IntTensor<B> {
        B::int_reshape(tensor, shape)
    }

    fn int_slice(tensor: IntTensor<B>, slices: &[Slice]) -> IntTensor<B> {
        B::int_slice(tensor, slices)
    }

    fn int_slice_assign(
        tensor: IntTensor<B>,
        slices: &[Slice],
        value: IntTensor<B>,
    ) -> IntTensor<B> {
        B::int_slice_assign(tensor, slices, value)
    }

    fn int_into_float(tensor: IntTensor<B>, dtype: FloatDType) -> Dual<B> {
        DualFloatTensor::constant(B::int_into_float(tensor, dtype))
    }

    fn int_mask_where(
        tensor: IntTensor<B>,
        mask: BoolTensor<B>,
        value: IntTensor<B>,
    ) -> IntTensor<B> {
        B::int_mask_where(tensor, mask, value)
    }

    fn int_mask_fill(tensor: IntTensor<B>, mask: BoolTensor<B>, value: Scalar) -> IntTensor<B> {
        B::int_mask_fill(tensor, mask, value)
    }

    fn int_gather(dim: usize, tensor: IntTensor<B>, indices: IntTensor<B>) -> IntTensor<B> {
        B::int_gather(dim, tensor, indices)
    }

    fn int_scatter(
        dim: usize,
        tensor: IntTensor<B>,
        indices: IntTensor<B>,
        value: IntTensor<B>,
        update: burn_backend::tensor::IndexingUpdateOp,
    ) -> IntTensor<B> {
        B::int_scatter(dim, tensor, indices, value, update)
    }

    fn int_select(tensor: IntTensor<B>, dim: usize, indices: IntTensor<B>) -> IntTensor<B> {
        B::int_select(tensor, dim, indices)
    }

    fn int_select_assign(
        tensor: IntTensor<B>,
        dim: usize,
        indices: IntTensor<B>,
        value: IntTensor<B>,
        update: burn_backend::tensor::IndexingUpdateOp,
    ) -> IntTensor<B> {
        B::int_select_assign(tensor, dim, indices, value, update)
    }

    fn int_equal(lhs: IntTensor<B>, rhs: IntTensor<B>, out_dtype: BoolStore) -> BoolTensor<B> {
        B::int_equal(lhs, rhs, out_dtype)
    }

    fn int_equal_elem(lhs: IntTensor<B>, rhs: Scalar, out_dtype: BoolStore) -> BoolTensor<B> {
        B::int_equal_elem(lhs, rhs, out_dtype)
    }

    fn int_greater(lhs: IntTensor<B>, rhs: IntTensor<B>, out_dtype: BoolStore) -> BoolTensor<B> {
        B::int_greater(lhs, rhs, out_dtype)
    }

    fn int_greater_elem(lhs: IntTensor<B>, rhs: Scalar, out_dtype: BoolStore) -> BoolTensor<B> {
        B::int_greater_elem(lhs, rhs, out_dtype)
    }

    fn int_greater_equal(
        lhs: IntTensor<B>,
        rhs: IntTensor<B>,
        out_dtype: BoolStore,
    ) -> BoolTensor<B> {
        B::int_greater_equal(lhs, rhs, out_dtype)
    }

    fn int_greater_equal_elem(
        lhs: IntTensor<B>,
        rhs: Scalar,
        out_dtype: BoolStore,
    ) -> BoolTensor<B> {
        B::int_greater_equal_elem(lhs, rhs, out_dtype)
    }

    fn int_lower(lhs: IntTensor<B>, rhs: IntTensor<B>, out_dtype: BoolStore) -> BoolTensor<B> {
        B::int_lower(lhs, rhs, out_dtype)
    }

    fn int_lower_elem(lhs: IntTensor<B>, rhs: Scalar, out_dtype: BoolStore) -> BoolTensor<B> {
        B::int_lower_elem(lhs, rhs, out_dtype)
    }

    fn int_lower_equal(
        lhs: IntTensor<B>,
        rhs: IntTensor<B>,
        out_dtype: BoolStore,
    ) -> BoolTensor<B> {
        B::int_lower_equal(lhs, rhs, out_dtype)
    }

    fn int_lower_equal_elem(lhs: IntTensor<B>, rhs: Scalar, out_dtype: BoolStore) -> BoolTensor<B> {
        B::int_lower_equal_elem(lhs, rhs, out_dtype)
    }

    fn int_add(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B> {
        B::int_add(lhs, rhs)
    }

    fn int_add_scalar(lhs: IntTensor<B>, rhs: Scalar) -> IntTensor<B> {
        B::int_add_scalar(lhs, rhs)
    }

    fn int_sub(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B> {
        B::int_sub(lhs, rhs)
    }

    fn int_sub_scalar(lhs: IntTensor<B>, rhs: Scalar) -> IntTensor<B> {
        B::int_sub_scalar(lhs, rhs)
    }

    fn int_mul(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B> {
        B::int_mul(lhs, rhs)
    }

    fn int_mul_scalar(lhs: IntTensor<B>, rhs: Scalar) -> IntTensor<B> {
        B::int_mul_scalar(lhs, rhs)
    }

    fn int_div(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B> {
        B::int_div(lhs, rhs)
    }

    fn int_div_scalar(lhs: IntTensor<B>, rhs: Scalar) -> IntTensor<B> {
        B::int_div_scalar(lhs, rhs)
    }

    fn int_remainder(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B> {
        B::int_remainder(lhs, rhs)
    }

    fn int_remainder_scalar(lhs: IntTensor<B>, rhs: Scalar) -> IntTensor<B> {
        B::int_remainder_scalar(lhs, rhs)
    }

    fn int_matmul(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B> {
        B::int_matmul(lhs, rhs)
    }

    fn int_sum(tensor: IntTensor<B>) -> IntTensor<B> {
        B::int_sum(tensor)
    }

    fn int_sum_dim(tensor: IntTensor<B>, dim: usize) -> IntTensor<B> {
        B::int_sum_dim(tensor, dim)
    }

    fn int_prod(tensor: IntTensor<B>) -> IntTensor<B> {
        B::int_prod(tensor)
    }

    fn int_prod_dim(tensor: IntTensor<B>, dim: usize) -> IntTensor<B> {
        B::int_prod_dim(tensor, dim)
    }

    fn int_mean_dim(tensor: IntTensor<B>, dim: usize) -> IntTensor<B> {
        B::int_mean_dim(tensor, dim)
    }

    fn int_cumsum(tensor: IntTensor<B>, dim: usize) -> IntTensor<B> {
        B::int_cumsum(tensor, dim)
    }

    fn int_cumprod(tensor: IntTensor<B>, dim: usize) -> IntTensor<B> {
        B::int_cumprod(tensor, dim)
    }

    fn int_cummin(tensor: IntTensor<B>, dim: usize) -> IntTensor<B> {
        B::int_cummin(tensor, dim)
    }

    fn int_cummax(tensor: IntTensor<B>, dim: usize) -> IntTensor<B> {
        B::int_cummax(tensor, dim)
    }

    fn int_argmax(tensor: IntTensor<B>, dim: usize) -> IntTensor<B> {
        B::int_argmax(tensor, dim)
    }

    fn int_argmin(tensor: IntTensor<B>, dim: usize) -> IntTensor<B> {
        B::int_argmin(tensor, dim)
    }

    fn int_abs(tensor: IntTensor<B>) -> IntTensor<B> {
        B::int_abs(tensor)
    }

    fn int_cast(tensor: IntTensor<B>, dtype: IntDType) -> IntTensor<B> {
        B::int_cast(tensor, dtype)
    }

    fn int_unfold(tensor: IntTensor<B>, dim: usize, size: usize, step: usize) -> IntTensor<B> {
        B::int_unfold(tensor, dim, size, step)
    }

    fn int_swap_dims(tensor: IntTensor<B>, dim1: usize, dim2: usize) -> IntTensor<B> {
        B::int_swap_dims(tensor, dim1, dim2)
    }

    fn int_permute(tensor: IntTensor<B>, axes: &[usize]) -> IntTensor<B> {
        B::int_permute(tensor, axes)
    }

    fn int_flip(tensor: IntTensor<B>, axes: &[usize]) -> IntTensor<B> {
        B::int_flip(tensor, axes)
    }

    fn int_random(
        shape: Shape,
        distribution: Distribution,
        device: &B::Device,
        dtype: IntDType,
    ) -> IntTensor<B> {
        B::int_random(shape, distribution, device, dtype)
    }

    fn int_expand(tensor: IntTensor<B>, shape: Shape) -> IntTensor<B> {
        B::int_expand(tensor, shape)
    }

    fn bitwise_and(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B> {
        B::bitwise_and(lhs, rhs)
    }

    fn bitwise_and_scalar(lhs: IntTensor<B>, rhs: Scalar) -> IntTensor<B> {
        B::bitwise_and_scalar(lhs, rhs)
    }

    fn bitwise_or(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B> {
        B::bitwise_or(lhs, rhs)
    }

    fn bitwise_or_scalar(lhs: IntTensor<B>, rhs: Scalar) -> IntTensor<B> {
        B::bitwise_or_scalar(lhs, rhs)
    }

    fn bitwise_xor(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B> {
        B::bitwise_xor(lhs, rhs)
    }

    fn bitwise_xor_scalar(lhs: IntTensor<B>, rhs: Scalar) -> IntTensor<B> {
        B::bitwise_xor_scalar(lhs, rhs)
    }

    fn bitwise_not(tensor: IntTensor<B>) -> IntTensor<B> {
        B::bitwise_not(tensor)
    }

    fn bitwise_left_shift(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B> {
        B::bitwise_left_shift(lhs, rhs)
    }

    fn bitwise_left_shift_scalar(lhs: IntTensor<B>, rhs: Scalar) -> IntTensor<B> {
        B::bitwise_left_shift_scalar(lhs, rhs)
    }

    fn bitwise_right_shift(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B> {
        B::bitwise_right_shift(lhs, rhs)
    }

    fn bitwise_right_shift_scalar(lhs: IntTensor<B>, rhs: Scalar) -> IntTensor<B> {
        B::bitwise_right_shift_scalar(lhs, rhs)
    }
}

impl<B: Backend> BoolTensorOps<ForwardDiff<B>> for ForwardDiff<B> {
    fn bool_empty(shape: Shape, device: &B::Device, dtype: BoolStore) -> BoolTensor<B> {
        B::bool_empty(shape, device, dtype)
    }

    fn bool_zeros(shape: Shape, device: &B::Device, dtype: BoolStore) -> BoolTensor<B> {
        B::bool_zeros(shape, device, dtype)
    }

    fn bool_ones(shape: Shape, device: &B::Device, dtype: BoolStore) -> BoolTensor<B> {
        B::bool_ones(shape, device, dtype)
    }

    fn bool_into_data(
        tensor: BoolTensor<B>,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send {
        B::bool_into_data(tensor)
    }

    fn bool_from_data(data: TensorData, device: &B::Device) -> BoolTensor<B> {
        B::bool_from_data(data, device)
    }

    fn bool_into_int(tensor: BoolTensor<B>, dtype: IntDType) -> IntTensor<B> {
        B::bool_into_int(tensor, dtype)
    }

    fn bool_into_float(tensor: BoolTensor<B>, dtype: FloatDType) -> Dual<B> {
        DualFloatTensor::constant(B::bool_into_float(tensor, dtype))
    }

    fn bool_to_device(tensor: BoolTensor<B>, device: &B::Device) -> BoolTensor<B> {
        B::bool_to_device(tensor, device)
    }

    fn bool_reshape(tensor: BoolTensor<B>, shape: Shape) -> BoolTensor<B> {
        B::bool_reshape(tensor, shape)
    }

    fn bool_slice(tensor: BoolTensor<B>, slices: &[Slice]) -> BoolTensor<B> {
        B::bool_slice(tensor, slices)
    }

    fn bool_slice_assign(
        tensor: BoolTensor<B>,
        slices: &[Slice],
        value: BoolTensor<B>,
    ) -> BoolTensor<B> {
        B::bool_slice_assign(tensor, slices, value)
    }

    fn bool_mask_where(
        tensor: BoolTensor<B>,
        mask: BoolTensor<B>,
        value: BoolTensor<B>,
    ) -> BoolTensor<B> {
        B::bool_mask_where(tensor, mask, value)
    }

    fn bool_mask_fill(tensor: BoolTensor<B>, mask: BoolTensor<B>, value: Scalar) -> BoolTensor<B> {
        B::bool_mask_fill(tensor, mask, value)
    }

    fn bool_gather(dim: usize, tensor: BoolTensor<B>, indices: IntTensor<B>) -> BoolTensor<B> {
        B::bool_gather(dim, tensor, indices)
    }

    fn bool_scatter(
        dim: usize,
        tensor: BoolTensor<B>,
        indices: IntTensor<B>,
        value: BoolTensor<B>,
        update: burn_backend::tensor::IndexingUpdateOp,
    ) -> BoolTensor<B> {
        B::bool_scatter(dim, tensor, indices, value, update)
    }

    fn bool_scatter_nd(
        tensor: BoolTensor<B>,
        indices: IntTensor<B>,
        value: BoolTensor<B>,
        update: burn_backend::tensor::IndexingUpdateOp,
    ) -> BoolTensor<B> {
        B::bool_scatter_nd(tensor, indices, value, update)
    }

    fn bool_gather_nd(tensor: BoolTensor<B>, indices: IntTensor<B>) -> BoolTensor<B> {
        B::bool_gather_nd(tensor, indices)
    }

    fn bool_select(tensor: BoolTensor<B>, dim: usize, indices: IntTensor<B>) -> BoolTensor<B> {
        B::bool_select(tensor, dim, indices)
    }

    fn bool_select_assign(
        tensor: BoolTensor<B>,
        dim: usize,
        indices: IntTensor<B>,
        value: BoolTensor<B>,
        update: burn_backend::tensor::IndexingUpdateOp,
    ) -> BoolTensor<B> {
        B::bool_select_assign(tensor, dim, indices, value, update)
    }

    fn bool_equal(lhs: BoolTensor<B>, rhs: BoolTensor<B>) -> BoolTensor<B> {
        B::bool_equal(lhs, rhs)
    }

    fn bool_equal_elem(lhs: BoolTensor<B>, rhs: Scalar) -> BoolTensor<B> {
        B::bool_equal_elem(lhs, rhs)
    }

    fn bool_not(tensor: BoolTensor<B>) -> BoolTensor<B> {
        B::bool_not(tensor)
    }

    fn bool_and(lhs: BoolTensor<B>, rhs: BoolTensor<B>) -> BoolTensor<B> {
        B::bool_and(lhs, rhs)
    }

    fn bool_or(lhs: BoolTensor<B>, rhs: BoolTensor<B>) -> BoolTensor<B> {
        B::bool_or(lhs, rhs)
    }

    fn bool_swap_dims(tensor: BoolTensor<B>, dim1: usize, dim2: usize) -> BoolTensor<B> {
        B::bool_swap_dims(tensor, dim1, dim2)
    }

    fn bool_permute(tensor: BoolTensor<B>, axes: &[usize]) -> BoolTensor<B> {
        B::bool_permute(tensor, axes)
    }

    fn bool_flip(tensor: BoolTensor<B>, axes: &[usize]) -> BoolTensor<B> {
        B::bool_flip(tensor, axes)
    }

    fn bool_expand(tensor: BoolTensor<B>, shape: Shape) -> BoolTensor<B> {
        B::bool_expand(tensor, shape)
    }

    fn bool_unfold(tensor: BoolTensor<B>, dim: usize, size: usize, step: usize) -> BoolTensor<B> {
        B::bool_unfold(tensor, dim, size, step)
    }
}

impl<B: Backend> QTensorOps<ForwardDiff<B>> for ForwardDiff<B> {
    fn q_from_data(
        data: TensorData,
        device: &B::Device,
    ) -> burn_backend::tensor::QuantizedTensor<B> {
        B::q_from_data(data, device)
    }

    fn quantize(
        tensor: Dual<B>,
        scheme: &QuantScheme,
        qparams: QuantizationParametersPrimitive<ForwardDiff<B>>,
    ) -> burn_backend::tensor::QuantizedTensor<B> {
        let qparams = QuantizationParametersPrimitive {
            scales: qparams.scales.primal,
            global: qparams.global.map(|global| global.primal),
        };
        B::quantize(tensor.primal, scheme, qparams)
    }

    fn dequantize(tensor: burn_backend::tensor::QuantizedTensor<B>, dtype: FloatDType) -> Dual<B> {
        DualFloatTensor::constant(B::dequantize(tensor, dtype))
    }

    fn q_to_device(
        tensor: burn_backend::tensor::QuantizedTensor<B>,
        device: &B::Device,
    ) -> burn_backend::tensor::QuantizedTensor<B> {
        B::q_to_device(tensor, device)
    }

    fn q_reshape(
        tensor: burn_backend::tensor::QuantizedTensor<B>,
        shape: Shape,
    ) -> burn_backend::tensor::QuantizedTensor<B> {
        B::q_reshape(tensor, shape)
    }

    fn q_into_data(
        tensor: burn_backend::tensor::QuantizedTensor<B>,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send {
        B::q_into_data(tensor)
    }

    fn q_swap_dims(
        tensor: burn_backend::tensor::QuantizedTensor<B>,
        dim1: usize,
        dim2: usize,
    ) -> burn_backend::tensor::QuantizedTensor<B> {
        B::q_swap_dims(tensor, dim1, dim2)
    }

    fn q_permute(
        tensor: burn_backend::tensor::QuantizedTensor<B>,
        axes: &[usize],
    ) -> burn_backend::tensor::QuantizedTensor<B> {
        B::q_permute(tensor, axes)
    }

    fn q_flip(
        tensor: burn_backend::tensor::QuantizedTensor<B>,
        axes: &[usize],
    ) -> burn_backend::tensor::QuantizedTensor<B> {
        B::q_flip(tensor, axes)
    }
}

impl<B: Backend> TransactionOps<ForwardDiff<B>> for ForwardDiff<B> {}

impl<B: Backend> DistributedOps<ForwardDiff<B>> for ForwardDiff<B> {}

#[cfg(test)]
mod tests {
    use crate::{ForwardDiff, FuncTensor, jvp};
    use alloc::vec;
    use alloc::vec::Vec;
    use burn_backend::TensorData;
    use burn_flex::{Flex, FlexDevice};

    type FD = ForwardDiff<Flex>;

    fn values(tensor: FuncTensor<Flex, 1>) -> Vec<f32> {
        tensor.into_data_sync().try_to_vec::<f32>().unwrap()
    }

    #[test]
    fn jvp_max_pool2d() {
        let device = FlexDevice;
        let x = FuncTensor::<Flex, 4>::from_data(
            TensorData::from([[[[1.0f32, 4.0], [2.0, 3.0]]]]),
            &device,
        );
        let dx = FuncTensor::<Flex, 4>::from_data(
            TensorData::from([[[[10.0f32, 20.0], [30.0, 40.0]]]]),
            &device,
        );

        let f = |x: &FuncTensor<FD, 4>| x.clone().max_pool2d([2, 2], [2, 2], [0, 0], [1, 1]);
        let tangent: FuncTensor<Flex, 4> = jvp::<Flex, _, _, _>(f, &x, &dx);

        let vals = tangent.into_data_sync().try_to_vec::<f32>().unwrap();
        // Max element is 4.0 at (0, 1), so winning tangent is 20.0
        assert_eq!(vals, vec![20.0]);
    }

    #[test]
    fn jvp_attention_smoke() {
        let device = FlexDevice;
        let q = FuncTensor::<Flex, 4>::from_data(
            TensorData::from([[[[1.0f32, 0.0], [0.0, 1.0]]]]),
            &device,
        );
        let k = FuncTensor::<Flex, 4>::from_data(
            TensorData::from([[[[1.0f32, 0.0], [0.0, 1.0]]]]),
            &device,
        );
        let v = FuncTensor::<Flex, 4>::from_data(
            TensorData::from([[[[1.0f32, 2.0], [3.0, 4.0]]]]),
            &device,
        );

        let dq = FuncTensor::<Flex, 4>::from_data(
            TensorData::from([[[[0.1f32, 0.0], [0.0, 0.0]]]]),
            &device,
        );
        let dk = FuncTensor::<Flex, 4>::zeros([1, 1, 2, 2], &device);
        let dv = FuncTensor::<Flex, 4>::zeros([1, 1, 2, 2], &device);

        let f = |(q, k, v): &(FuncTensor<FD, 4>, FuncTensor<FD, 4>, FuncTensor<FD, 4>)| {
            q.clone().attention(k.clone(), v.clone()).sum()
        };
        let tangent: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(f, &(q, k, v), &(dq, dk, dv));
        let val = values(tangent)[0];
        assert!(val.is_finite());
        assert!(val.abs() > 1e-4);
    }

    #[test]
    fn jvp_grid_sample_2d_smoke() {
        use burn_backend::ops::{GridSampleOptions, InterpolateMode};
        let device = FlexDevice;
        let tensor = FuncTensor::<Flex, 4>::from_data(
            TensorData::from([[[[1.0f32, 2.0], [3.0, 4.0]]]]),
            &device,
        );
        let grid = FuncTensor::<Flex, 4>::from_data(
            TensorData::from([[[[-0.5f32, -0.5], [0.5, -0.5]], [[-0.5, 0.5], [0.5, 0.5]]]]),
            &device,
        );
        let dt = FuncTensor::<Flex, 4>::from_data(
            TensorData::from([[[[1.0f32, 1.0], [1.0, 1.0]]]]),
            &device,
        );
        let dg = FuncTensor::<Flex, 4>::zeros([1, 2, 2, 2], &device);

        let options = GridSampleOptions::new(InterpolateMode::Bilinear);
        let f = |(t, g): &(FuncTensor<FD, 4>, FuncTensor<FD, 4>)| {
            t.clone().grid_sample_2d(g.clone(), options.clone()).sum()
        };
        let tangent: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(f, &(tensor, grid), &(dt, dg));
        let val = values(tangent)[0];
        assert!(val.is_finite());
        assert!((val - 4.0).abs() < 1e-3);
    }
}
