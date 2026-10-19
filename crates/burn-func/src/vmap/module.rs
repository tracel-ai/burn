use alloc::vec::Vec;
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

use super::{
    BatchedPrimitive, Vmap, batched_axis, prepend_full_batch_slice, promote_bool, promote_float,
    promote_int, unify_bool, unify_int,
};

type Batched<P> = BatchedPrimitive<P>;
type BatchedFloat<B> = Batched<FloatTensor<B>>;
type BatchedInt<B> = Batched<IntTensor<B>>;
type BatchedBool<B> = Batched<BoolTensor<B>>;

fn fold_batch_dim_float<B: Backend>(
    tensor: FloatTensor<B>,
    vmap_n: usize,
) -> (FloatTensor<B>, usize) {
    let shape = tensor.shape();
    let dims = shape.as_slice();
    assert!(
        dims.len() >= 2,
        "Tensor must have at least 2 dimensions to fold batch axis"
    );
    assert_eq!(dims[0], vmap_n, "Leading dimension must match vmap_n");
    let n_batch = dims[1];
    let mut folded_dims = Vec::with_capacity(dims.len() - 1);
    folded_dims.push(vmap_n * n_batch);
    folded_dims.extend_from_slice(&dims[2..]);
    (B::float_reshape(tensor, Shape::from(folded_dims)), n_batch)
}

fn unfold_batch_dim_float<B: Backend>(
    tensor: FloatTensor<B>,
    vmap_n: usize,
    n_batch: usize,
) -> FloatTensor<B> {
    let shape = tensor.shape();
    let dims = shape.as_slice();
    let mut unfolded_dims = Vec::with_capacity(dims.len() + 1);
    unfolded_dims.push(vmap_n);
    unfolded_dims.push(n_batch);
    unfolded_dims.extend_from_slice(&dims[1..]);
    B::float_reshape(tensor, Shape::from(unfolded_dims))
}

fn fold_batch_dim_int<B: Backend>(tensor: IntTensor<B>, vmap_n: usize) -> (IntTensor<B>, usize) {
    let shape = tensor.shape();
    let dims = shape.as_slice();
    assert!(
        dims.len() >= 2,
        "Tensor must have at least 2 dimensions to fold batch axis"
    );
    assert_eq!(dims[0], vmap_n, "Leading dimension must match vmap_n");
    let n_batch = dims[1];
    let mut folded_dims = Vec::with_capacity(dims.len() - 1);
    folded_dims.push(vmap_n * n_batch);
    folded_dims.extend_from_slice(&dims[2..]);
    (B::int_reshape(tensor, Shape::from(folded_dims)), n_batch)
}

fn unfold_batch_dim_int<B: Backend>(
    tensor: IntTensor<B>,
    vmap_n: usize,
    n_batch: usize,
) -> IntTensor<B> {
    let shape = tensor.shape();
    let dims = shape.as_slice();
    let mut unfolded_dims = Vec::with_capacity(dims.len() + 1);
    unfolded_dims.push(vmap_n);
    unfolded_dims.push(n_batch);
    unfolded_dims.extend_from_slice(&dims[1..]);
    B::int_reshape(tensor, Shape::from(unfolded_dims))
}

fn fold_batch_dim_bool<B: Backend>(tensor: BoolTensor<B>, vmap_n: usize) -> (BoolTensor<B>, usize) {
    let shape = tensor.shape();
    let dims = shape.as_slice();
    assert!(
        dims.len() >= 2,
        "Tensor must have at least 2 dimensions to fold batch axis"
    );
    assert_eq!(dims[0], vmap_n, "Leading dimension must match vmap_n");
    let n_batch = dims[1];
    let mut folded_dims = Vec::with_capacity(dims.len() - 1);
    folded_dims.push(vmap_n * n_batch);
    folded_dims.extend_from_slice(&dims[2..]);
    (B::bool_reshape(tensor, Shape::from(folded_dims)), n_batch)
}

impl<B: Backend> ActivationOps<Vmap<B>> for Vmap<B> {}

impl<B: Backend> ModuleOps<Vmap<B>> for Vmap<B> {
    fn conv2d(
        x: BatchedFloat<B>,
        weight: BatchedFloat<B>,
        bias: Option<BatchedFloat<B>>,
        options: ConvOptions<2>,
    ) -> BatchedFloat<B> {
        match x.batch_size {
            Some(vmap_n) => {
                let (folded_x, n_batch) = fold_batch_dim_float::<B>(x.primitive, vmap_n);
                let out = B::conv2d(
                    folded_x,
                    weight.primitive,
                    bias.map(|bias| bias.primitive),
                    options,
                );
                let unfolded = unfold_batch_dim_float::<B>(out, vmap_n, n_batch);
                BatchedPrimitive::batched(unfolded, vmap_n)
            }
            None => BatchedPrimitive::unbatched(B::conv2d(
                x.primitive,
                weight.primitive,
                bias.map(|bias| bias.primitive),
                options,
            )),
        }
    }

    fn deform_conv2d(
        x: BatchedFloat<B>,
        offset: BatchedFloat<B>,
        weight: BatchedFloat<B>,
        mask: Option<BatchedFloat<B>>,
        bias: Option<BatchedFloat<B>>,
        options: DeformConvOptions<2>,
    ) -> BatchedFloat<B> {
        BatchedPrimitive {
            primitive: B::deform_conv2d(
                x.primitive,
                offset.primitive,
                weight.primitive,
                mask.map(|mask| mask.primitive),
                bias.map(|bias| bias.primitive),
                options,
            ),
            batch_size: x.batch_size,
        }
    }

    fn deform_conv2d_backward(
        x: BatchedFloat<B>,
        offset: BatchedFloat<B>,
        weight: BatchedFloat<B>,
        mask: Option<BatchedFloat<B>>,
        bias: Option<BatchedFloat<B>>,
        output_grad: BatchedFloat<B>,
        options: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<Vmap<B>> {
        let result = B::deform_conv2d_backward(
            x.primitive,
            offset.primitive,
            weight.primitive,
            mask.map(|mask| mask.primitive),
            bias.map(|bias| bias.primitive),
            output_grad.primitive,
            options,
        );
        DeformConv2dBackward::new(
            BatchedPrimitive::unbatched(result.x_grad),
            BatchedPrimitive::unbatched(result.offset_grad),
            BatchedPrimitive::unbatched(result.weight_grad),
            result.mask_grad.map(BatchedPrimitive::unbatched),
            result.bias_grad.map(BatchedPrimitive::unbatched),
        )
    }

    fn conv3d(
        x: BatchedFloat<B>,
        weight: BatchedFloat<B>,
        bias: Option<BatchedFloat<B>>,
        options: ConvOptions<3>,
    ) -> BatchedFloat<B> {
        match x.batch_size {
            Some(vmap_n) => {
                let (folded_x, n_batch) = fold_batch_dim_float::<B>(x.primitive, vmap_n);
                let out = B::conv3d(
                    folded_x,
                    weight.primitive,
                    bias.map(|bias| bias.primitive),
                    options,
                );
                let unfolded = unfold_batch_dim_float::<B>(out, vmap_n, n_batch);
                BatchedPrimitive::batched(unfolded, vmap_n)
            }
            None => BatchedPrimitive::unbatched(B::conv3d(
                x.primitive,
                weight.primitive,
                bias.map(|bias| bias.primitive),
                options,
            )),
        }
    }

    fn conv_transpose2d(
        x: BatchedFloat<B>,
        weight: BatchedFloat<B>,
        bias: Option<BatchedFloat<B>>,
        options: ConvTransposeOptions<2>,
    ) -> BatchedFloat<B> {
        match x.batch_size {
            Some(vmap_n) => {
                let (folded_x, n_batch) = fold_batch_dim_float::<B>(x.primitive, vmap_n);
                let out = B::conv_transpose2d(
                    folded_x,
                    weight.primitive,
                    bias.map(|bias| bias.primitive),
                    options,
                );
                let unfolded = unfold_batch_dim_float::<B>(out, vmap_n, n_batch);
                BatchedPrimitive::batched(unfolded, vmap_n)
            }
            None => BatchedPrimitive::unbatched(B::conv_transpose2d(
                x.primitive,
                weight.primitive,
                bias.map(|bias| bias.primitive),
                options,
            )),
        }
    }

    fn conv_transpose3d(
        x: BatchedFloat<B>,
        weight: BatchedFloat<B>,
        bias: Option<BatchedFloat<B>>,
        options: ConvTransposeOptions<3>,
    ) -> BatchedFloat<B> {
        match x.batch_size {
            Some(vmap_n) => {
                let (folded_x, n_batch) = fold_batch_dim_float::<B>(x.primitive, vmap_n);
                let out = B::conv_transpose3d(
                    folded_x,
                    weight.primitive,
                    bias.map(|bias| bias.primitive),
                    options,
                );
                let unfolded = unfold_batch_dim_float::<B>(out, vmap_n, n_batch);
                BatchedPrimitive::batched(unfolded, vmap_n)
            }
            None => BatchedPrimitive::unbatched(B::conv_transpose3d(
                x.primitive,
                weight.primitive,
                bias.map(|bias| bias.primitive),
                options,
            )),
        }
    }

    fn avg_pool2d(
        x: BatchedFloat<B>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> BatchedFloat<B> {
        match x.batch_size {
            Some(vmap_n) => {
                let (folded_x, n_batch) = fold_batch_dim_float::<B>(x.primitive, vmap_n);
                let out = B::avg_pool2d(
                    folded_x,
                    kernel_size,
                    stride,
                    padding,
                    count_include_pad,
                    ceil_mode,
                );
                let unfolded = unfold_batch_dim_float::<B>(out, vmap_n, n_batch);
                BatchedPrimitive::batched(unfolded, vmap_n)
            }
            None => BatchedPrimitive::unbatched(B::avg_pool2d(
                x.primitive,
                kernel_size,
                stride,
                padding,
                count_include_pad,
                ceil_mode,
            )),
        }
    }

    fn avg_pool2d_backward(
        x: BatchedFloat<B>,
        grad: BatchedFloat<B>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> BatchedFloat<B> {
        let batch_size = x.batch_size.or(grad.batch_size);
        match batch_size {
            Some(vmap_n) => {
                let x_prim = promote_float::<B>(x, vmap_n);
                let grad_prim = promote_float::<B>(grad, vmap_n);
                let (folded_x, n_batch) = fold_batch_dim_float::<B>(x_prim, vmap_n);
                let (folded_grad, _) = fold_batch_dim_float::<B>(grad_prim, vmap_n);
                let out = B::avg_pool2d_backward(
                    folded_x,
                    folded_grad,
                    kernel_size,
                    stride,
                    padding,
                    count_include_pad,
                    ceil_mode,
                );
                let unfolded = unfold_batch_dim_float::<B>(out, vmap_n, n_batch);
                BatchedPrimitive::batched(unfolded, vmap_n)
            }
            None => BatchedPrimitive::unbatched(B::avg_pool2d_backward(
                x.primitive,
                grad.primitive,
                kernel_size,
                stride,
                padding,
                count_include_pad,
                ceil_mode,
            )),
        }
    }

    fn avg_pool3d(
        x: BatchedFloat<B>,
        kernel_size: [usize; 3],
        stride: [usize; 3],
        padding: [usize; 3],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> BatchedFloat<B> {
        match x.batch_size {
            Some(vmap_n) => {
                let (folded_x, n_batch) = fold_batch_dim_float::<B>(x.primitive, vmap_n);
                let out = B::avg_pool3d(
                    folded_x,
                    kernel_size,
                    stride,
                    padding,
                    count_include_pad,
                    ceil_mode,
                );
                let unfolded = unfold_batch_dim_float::<B>(out, vmap_n, n_batch);
                BatchedPrimitive::batched(unfolded, vmap_n)
            }
            None => BatchedPrimitive::unbatched(B::avg_pool3d(
                x.primitive,
                kernel_size,
                stride,
                padding,
                count_include_pad,
                ceil_mode,
            )),
        }
    }

    fn avg_pool3d_backward(
        x: BatchedFloat<B>,
        grad: BatchedFloat<B>,
        kernel_size: [usize; 3],
        stride: [usize; 3],
        padding: [usize; 3],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> BatchedFloat<B> {
        let batch_size = x.batch_size.or(grad.batch_size);
        match batch_size {
            Some(vmap_n) => {
                let x_prim = promote_float::<B>(x, vmap_n);
                let grad_prim = promote_float::<B>(grad, vmap_n);
                let (folded_x, n_batch) = fold_batch_dim_float::<B>(x_prim, vmap_n);
                let (folded_grad, _) = fold_batch_dim_float::<B>(grad_prim, vmap_n);
                let out = B::avg_pool3d_backward(
                    folded_x,
                    folded_grad,
                    kernel_size,
                    stride,
                    padding,
                    count_include_pad,
                    ceil_mode,
                );
                let unfolded = unfold_batch_dim_float::<B>(out, vmap_n, n_batch);
                BatchedPrimitive::batched(unfolded, vmap_n)
            }
            None => BatchedPrimitive::unbatched(B::avg_pool3d_backward(
                x.primitive,
                grad.primitive,
                kernel_size,
                stride,
                padding,
                count_include_pad,
                ceil_mode,
            )),
        }
    }

    fn adaptive_avg_pool2d(x: BatchedFloat<B>, output_size: [usize; 2]) -> BatchedFloat<B> {
        match x.batch_size {
            Some(vmap_n) => {
                let (folded_x, n_batch) = fold_batch_dim_float::<B>(x.primitive, vmap_n);
                let out = B::adaptive_avg_pool2d(folded_x, output_size);
                let unfolded = unfold_batch_dim_float::<B>(out, vmap_n, n_batch);
                BatchedPrimitive::batched(unfolded, vmap_n)
            }
            None => BatchedPrimitive::unbatched(B::adaptive_avg_pool2d(x.primitive, output_size)),
        }
    }

    fn adaptive_avg_pool2d_backward(x: BatchedFloat<B>, grad: BatchedFloat<B>) -> BatchedFloat<B> {
        let batch_size = x.batch_size.or(grad.batch_size);
        match batch_size {
            Some(vmap_n) => {
                let x_prim = promote_float::<B>(x, vmap_n);
                let grad_prim = promote_float::<B>(grad, vmap_n);
                let (folded_x, n_batch) = fold_batch_dim_float::<B>(x_prim, vmap_n);
                let (folded_grad, _) = fold_batch_dim_float::<B>(grad_prim, vmap_n);
                let out = B::adaptive_avg_pool2d_backward(folded_x, folded_grad);
                let unfolded = unfold_batch_dim_float::<B>(out, vmap_n, n_batch);
                BatchedPrimitive::batched(unfolded, vmap_n)
            }
            None => BatchedPrimitive::unbatched(B::adaptive_avg_pool2d_backward(
                x.primitive,
                grad.primitive,
            )),
        }
    }

    fn adaptive_avg_pool3d(x: BatchedFloat<B>, output_size: [usize; 3]) -> BatchedFloat<B> {
        match x.batch_size {
            Some(vmap_n) => {
                let (folded_x, n_batch) = fold_batch_dim_float::<B>(x.primitive, vmap_n);
                let out = B::adaptive_avg_pool3d(folded_x, output_size);
                let unfolded = unfold_batch_dim_float::<B>(out, vmap_n, n_batch);
                BatchedPrimitive::batched(unfolded, vmap_n)
            }
            None => BatchedPrimitive::unbatched(B::adaptive_avg_pool3d(x.primitive, output_size)),
        }
    }

    fn adaptive_avg_pool3d_backward(x: BatchedFloat<B>, grad: BatchedFloat<B>) -> BatchedFloat<B> {
        let batch_size = x.batch_size.or(grad.batch_size);
        match batch_size {
            Some(vmap_n) => {
                let x_prim = promote_float::<B>(x, vmap_n);
                let grad_prim = promote_float::<B>(grad, vmap_n);
                let (folded_x, n_batch) = fold_batch_dim_float::<B>(x_prim, vmap_n);
                let (folded_grad, _) = fold_batch_dim_float::<B>(grad_prim, vmap_n);
                let out = B::adaptive_avg_pool3d_backward(folded_x, folded_grad);
                let unfolded = unfold_batch_dim_float::<B>(out, vmap_n, n_batch);
                BatchedPrimitive::batched(unfolded, vmap_n)
            }
            None => BatchedPrimitive::unbatched(B::adaptive_avg_pool3d_backward(
                x.primitive,
                grad.primitive,
            )),
        }
    }

    fn max_pool2d(
        x: BatchedFloat<B>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> BatchedFloat<B> {
        match x.batch_size {
            Some(vmap_n) => {
                let (folded_x, n_batch) = fold_batch_dim_float::<B>(x.primitive, vmap_n);
                let out =
                    B::max_pool2d(folded_x, kernel_size, stride, padding, dilation, ceil_mode);
                let unfolded = unfold_batch_dim_float::<B>(out, vmap_n, n_batch);
                BatchedPrimitive::batched(unfolded, vmap_n)
            }
            None => BatchedPrimitive::unbatched(B::max_pool2d(
                x.primitive,
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
            )),
        }
    }

    fn max_pool2d_with_indices(
        x: BatchedFloat<B>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
        indices_dtype: IntDType,
    ) -> MaxPool2dWithIndices<Vmap<B>> {
        match x.batch_size {
            Some(vmap_n) => {
                let (folded_x, n_batch) = fold_batch_dim_float::<B>(x.primitive, vmap_n);
                let result = B::max_pool2d_with_indices(
                    folded_x,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    ceil_mode,
                    indices_dtype,
                );
                let unfolded_out = unfold_batch_dim_float::<B>(result.output, vmap_n, n_batch);
                let unfolded_indices = unfold_batch_dim_int::<B>(result.indices, vmap_n, n_batch);
                MaxPool2dWithIndices::new(
                    BatchedPrimitive::batched(unfolded_out, vmap_n),
                    BatchedPrimitive::batched(unfolded_indices, vmap_n),
                )
            }
            None => {
                let result = B::max_pool2d_with_indices(
                    x.primitive,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    ceil_mode,
                    indices_dtype,
                );
                MaxPool2dWithIndices::new(
                    BatchedPrimitive::unbatched(result.output),
                    BatchedPrimitive::unbatched(result.indices),
                )
            }
        }
    }

    fn max_pool2d_with_indices_backward(
        x: BatchedFloat<B>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
        output_grad: BatchedFloat<B>,
        indices: BatchedInt<B>,
    ) -> MaxPool2dBackward<Vmap<B>> {
        let batch_size = x
            .batch_size
            .or(output_grad.batch_size)
            .or(indices.batch_size);
        match batch_size {
            Some(vmap_n) => {
                let x_prim = promote_float::<B>(x, vmap_n);
                let out_grad_prim = promote_float::<B>(output_grad, vmap_n);
                let indices_prim = promote_int::<B>(indices, vmap_n);
                let (folded_x, n_batch) = fold_batch_dim_float::<B>(x_prim, vmap_n);
                let (folded_out_grad, _) = fold_batch_dim_float::<B>(out_grad_prim, vmap_n);
                let (folded_indices, _) = fold_batch_dim_int::<B>(indices_prim, vmap_n);
                let result = B::max_pool2d_with_indices_backward(
                    folded_x,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    ceil_mode,
                    folded_out_grad,
                    folded_indices,
                );
                let unfolded = unfold_batch_dim_float::<B>(result.x_grad, vmap_n, n_batch);
                MaxPool2dBackward::new(BatchedPrimitive::batched(unfolded, vmap_n))
            }
            None => {
                let result = B::max_pool2d_with_indices_backward(
                    x.primitive,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    ceil_mode,
                    output_grad.primitive,
                    indices.primitive,
                );
                MaxPool2dBackward::new(BatchedPrimitive::unbatched(result.x_grad))
            }
        }
    }

    fn max_pool3d(
        x: BatchedFloat<B>,
        kernel_size: [usize; 3],
        stride: [usize; 3],
        padding: [usize; 3],
        dilation: [usize; 3],
        ceil_mode: bool,
    ) -> BatchedFloat<B> {
        match x.batch_size {
            Some(vmap_n) => {
                let (folded_x, n_batch) = fold_batch_dim_float::<B>(x.primitive, vmap_n);
                let out =
                    B::max_pool3d(folded_x, kernel_size, stride, padding, dilation, ceil_mode);
                let unfolded = unfold_batch_dim_float::<B>(out, vmap_n, n_batch);
                BatchedPrimitive::batched(unfolded, vmap_n)
            }
            None => BatchedPrimitive::unbatched(B::max_pool3d(
                x.primitive,
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
            )),
        }
    }

    fn max_pool3d_with_indices(
        x: BatchedFloat<B>,
        kernel_size: [usize; 3],
        stride: [usize; 3],
        padding: [usize; 3],
        dilation: [usize; 3],
        ceil_mode: bool,
        indices_dtype: IntDType,
    ) -> MaxPool3dWithIndices<Vmap<B>> {
        match x.batch_size {
            Some(vmap_n) => {
                let (folded_x, n_batch) = fold_batch_dim_float::<B>(x.primitive, vmap_n);
                let result = B::max_pool3d_with_indices(
                    folded_x,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    ceil_mode,
                    indices_dtype,
                );
                let unfolded_out = unfold_batch_dim_float::<B>(result.output, vmap_n, n_batch);
                let unfolded_indices = unfold_batch_dim_int::<B>(result.indices, vmap_n, n_batch);
                MaxPool3dWithIndices::new(
                    BatchedPrimitive::batched(unfolded_out, vmap_n),
                    BatchedPrimitive::batched(unfolded_indices, vmap_n),
                )
            }
            None => {
                let result = B::max_pool3d_with_indices(
                    x.primitive,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    ceil_mode,
                    indices_dtype,
                );
                MaxPool3dWithIndices::new(
                    BatchedPrimitive::unbatched(result.output),
                    BatchedPrimitive::unbatched(result.indices),
                )
            }
        }
    }

    fn max_pool3d_with_indices_backward(
        x: BatchedFloat<B>,
        kernel_size: [usize; 3],
        stride: [usize; 3],
        padding: [usize; 3],
        dilation: [usize; 3],
        ceil_mode: bool,
        output_grad: BatchedFloat<B>,
        indices: BatchedInt<B>,
    ) -> MaxPool3dBackward<Vmap<B>> {
        let batch_size = x
            .batch_size
            .or(output_grad.batch_size)
            .or(indices.batch_size);
        match batch_size {
            Some(vmap_n) => {
                let x_prim = promote_float::<B>(x, vmap_n);
                let out_grad_prim = promote_float::<B>(output_grad, vmap_n);
                let indices_prim = promote_int::<B>(indices, vmap_n);
                let (folded_x, n_batch) = fold_batch_dim_float::<B>(x_prim, vmap_n);
                let (folded_out_grad, _) = fold_batch_dim_float::<B>(out_grad_prim, vmap_n);
                let (folded_indices, _) = fold_batch_dim_int::<B>(indices_prim, vmap_n);
                let result = B::max_pool3d_with_indices_backward(
                    folded_x,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    ceil_mode,
                    folded_out_grad,
                    folded_indices,
                );
                let unfolded = unfold_batch_dim_float::<B>(result.x_grad, vmap_n, n_batch);
                MaxPool3dBackward::new(BatchedPrimitive::batched(unfolded, vmap_n))
            }
            None => {
                let result = B::max_pool3d_with_indices_backward(
                    x.primitive,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    ceil_mode,
                    output_grad.primitive,
                    indices.primitive,
                );
                MaxPool3dBackward::new(BatchedPrimitive::unbatched(result.x_grad))
            }
        }
    }

    fn interpolate(
        x: BatchedFloat<B>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> BatchedFloat<B> {
        match x.batch_size {
            Some(vmap_n) => {
                let (folded_x, n_batch) = fold_batch_dim_float::<B>(x.primitive, vmap_n);
                let out = B::interpolate(folded_x, output_size, options);
                let unfolded = unfold_batch_dim_float::<B>(out, vmap_n, n_batch);
                BatchedPrimitive::batched(unfolded, vmap_n)
            }
            None => BatchedPrimitive::unbatched(B::interpolate(x.primitive, output_size, options)),
        }
    }

    fn interpolate_backward(
        x: BatchedFloat<B>,
        grad: BatchedFloat<B>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> BatchedFloat<B> {
        let batch_size = x.batch_size.or(grad.batch_size);
        match batch_size {
            Some(vmap_n) => {
                let x_prim = promote_float::<B>(x, vmap_n);
                let grad_prim = promote_float::<B>(grad, vmap_n);
                let (folded_x, n_batch) = fold_batch_dim_float::<B>(x_prim, vmap_n);
                let (folded_grad, _) = fold_batch_dim_float::<B>(grad_prim, vmap_n);
                let out = B::interpolate_backward(folded_x, folded_grad, output_size, options);
                let unfolded = unfold_batch_dim_float::<B>(out, vmap_n, n_batch);
                BatchedPrimitive::batched(unfolded, vmap_n)
            }
            None => BatchedPrimitive::unbatched(B::interpolate_backward(
                x.primitive,
                grad.primitive,
                output_size,
                options,
            )),
        }
    }

    fn attention(
        query: BatchedFloat<B>,
        key: BatchedFloat<B>,
        value: BatchedFloat<B>,
        mask: Option<BatchedBool<B>>,
        attn_bias: Option<BatchedFloat<B>>,
        options: AttentionModuleOptions,
    ) -> BatchedFloat<B> {
        let batch_size = query
            .batch_size
            .or(key.batch_size)
            .or(value.batch_size)
            .or(mask.as_ref().and_then(|m| m.batch_size))
            .or(attn_bias.as_ref().and_then(|b| b.batch_size));
        match batch_size {
            Some(vmap_n) => {
                let query_prim = promote_float::<B>(query, vmap_n);
                let key_prim = promote_float::<B>(key, vmap_n);
                let value_prim = promote_float::<B>(value, vmap_n);
                let (folded_q, n_batch) = fold_batch_dim_float::<B>(query_prim, vmap_n);
                let (folded_k, _) = fold_batch_dim_float::<B>(key_prim, vmap_n);
                let (folded_v, _) = fold_batch_dim_float::<B>(value_prim, vmap_n);
                let folded_mask = mask.map(|m| {
                    let m_prim = promote_bool::<B>(m, vmap_n);
                    let (f_m, _) = fold_batch_dim_bool::<B>(m_prim, vmap_n);
                    f_m
                });
                let folded_bias = attn_bias.map(|b| {
                    let b_prim = promote_float::<B>(b, vmap_n);
                    let (f_b, _) = fold_batch_dim_float::<B>(b_prim, vmap_n);
                    f_b
                });
                let out = B::attention(
                    folded_q,
                    folded_k,
                    folded_v,
                    folded_mask,
                    folded_bias,
                    options,
                );
                let unfolded = unfold_batch_dim_float::<B>(out, vmap_n, n_batch);
                BatchedPrimitive::batched(unfolded, vmap_n)
            }
            None => BatchedPrimitive::unbatched(B::attention(
                query.primitive,
                key.primitive,
                value.primitive,
                mask.map(|m| m.primitive),
                attn_bias.map(|b| b.primitive),
                options,
            )),
        }
    }
}

macro_rules! int_binary {
    ($name:ident, $backend:ident) => {
        fn $name(lhs: BatchedInt<B>, rhs: BatchedInt<B>) -> BatchedInt<B> {
            let (lhs, rhs, batch_size) = unify_int::<B>(lhs, rhs);
            BatchedPrimitive {
                primitive: B::$backend(lhs, rhs),
                batch_size,
            }
        }
    };
}

macro_rules! int_binary_scalar {
    ($name:ident, $backend:ident) => {
        fn $name(lhs: BatchedInt<B>, rhs: Scalar) -> BatchedInt<B> {
            BatchedPrimitive {
                primitive: B::$backend(lhs.primitive, rhs),
                batch_size: lhs.batch_size,
            }
        }
    };
}

macro_rules! int_unary {
    ($name:ident, $backend:ident) => {
        fn $name(tensor: BatchedInt<B>) -> BatchedInt<B> {
            BatchedPrimitive {
                primitive: B::$backend(tensor.primitive),
                batch_size: tensor.batch_size,
            }
        }
    };
}

macro_rules! int_dim {
    ($name:ident, $backend:ident) => {
        fn $name(tensor: BatchedInt<B>, dim: usize) -> BatchedInt<B> {
            BatchedPrimitive {
                primitive: B::$backend(tensor.primitive, batched_axis(tensor.batch_size, dim)),
                batch_size: tensor.batch_size,
            }
        }
    };
}

impl<B: Backend> IntTensorOps<Vmap<B>> for Vmap<B> {
    fn int_empty(shape: Shape, device: &B::Device, dtype: IntDType) -> BatchedInt<B> {
        BatchedPrimitive::unbatched(B::int_empty(shape, device, dtype))
    }

    fn int_into_data(
        tensor: BatchedInt<B>,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send {
        B::int_into_data(tensor.primitive)
    }

    fn int_from_data(data: TensorData, device: &B::Device) -> BatchedInt<B> {
        BatchedPrimitive::unbatched(B::int_from_data(data, device))
    }

    fn int_to_device(tensor: BatchedInt<B>, device: &B::Device) -> BatchedInt<B> {
        BatchedPrimitive {
            primitive: B::int_to_device(tensor.primitive, device),
            batch_size: tensor.batch_size,
        }
    }

    fn int_reshape(tensor: BatchedInt<B>, shape: Shape) -> BatchedInt<B> {
        let shape = super::batched_shape(tensor.batch_size, &shape);
        BatchedPrimitive {
            primitive: B::int_reshape(tensor.primitive, shape),
            batch_size: tensor.batch_size,
        }
    }

    fn int_slice(tensor: BatchedInt<B>, slices: &[Slice]) -> BatchedInt<B> {
        let slices = prepend_full_batch_slice(tensor.batch_size, slices);
        BatchedPrimitive {
            primitive: B::int_slice(tensor.primitive, &slices),
            batch_size: tensor.batch_size,
        }
    }

    fn int_slice_assign(
        tensor: BatchedInt<B>,
        slices: &[Slice],
        value: BatchedInt<B>,
    ) -> BatchedInt<B> {
        let batch_size = tensor.batch_size.or(value.batch_size);
        let (tensor, value) = match batch_size {
            Some(n) => (promote_int::<B>(tensor, n), promote_int::<B>(value, n)),
            None => (tensor.primitive, value.primitive),
        };
        let slices = prepend_full_batch_slice(batch_size, slices);
        BatchedPrimitive {
            primitive: B::int_slice_assign(tensor, &slices, value),
            batch_size,
        }
    }

    fn int_into_float(tensor: BatchedInt<B>, dtype: FloatDType) -> BatchedFloat<B> {
        BatchedPrimitive {
            primitive: B::int_into_float(tensor.primitive, dtype),
            batch_size: tensor.batch_size,
        }
    }

    fn int_mask_where(
        tensor: BatchedInt<B>,
        mask: BatchedBool<B>,
        value: BatchedInt<B>,
    ) -> BatchedInt<B> {
        let batch_size = tensor.batch_size.or(mask.batch_size).or(value.batch_size);
        match batch_size {
            Some(n) => BatchedPrimitive::batched(
                B::int_mask_where(
                    promote_int::<B>(tensor, n),
                    promote_bool::<B>(mask, n),
                    promote_int::<B>(value, n),
                ),
                n,
            ),
            None => BatchedPrimitive::unbatched(B::int_mask_where(
                tensor.primitive,
                mask.primitive,
                value.primitive,
            )),
        }
    }

    fn int_mask_fill(tensor: BatchedInt<B>, mask: BatchedBool<B>, value: Scalar) -> BatchedInt<B> {
        let batch_size = tensor.batch_size.or(mask.batch_size);
        match batch_size {
            Some(n) => BatchedPrimitive::batched(
                B::int_mask_fill(
                    promote_int::<B>(tensor, n),
                    promote_bool::<B>(mask, n),
                    value,
                ),
                n,
            ),
            None => BatchedPrimitive::unbatched(B::int_mask_fill(
                tensor.primitive,
                mask.primitive,
                value,
            )),
        }
    }

    fn int_gather(dim: usize, tensor: BatchedInt<B>, indices: BatchedInt<B>) -> BatchedInt<B> {
        let batch_size = tensor.batch_size.or(indices.batch_size);
        match batch_size {
            Some(n) => BatchedPrimitive::batched(
                B::int_gather(
                    dim + 1,
                    promote_int::<B>(tensor, n),
                    promote_int::<B>(indices, n),
                ),
                n,
            ),
            None => {
                BatchedPrimitive::unbatched(B::int_gather(dim, tensor.primitive, indices.primitive))
            }
        }
    }

    fn int_scatter(
        dim: usize,
        tensor: BatchedInt<B>,
        indices: BatchedInt<B>,
        value: BatchedInt<B>,
        update: burn_backend::tensor::IndexingUpdateOp,
    ) -> BatchedInt<B> {
        let batch_size = tensor
            .batch_size
            .or(indices.batch_size)
            .or(value.batch_size);
        match batch_size {
            Some(n) => BatchedPrimitive::batched(
                B::int_scatter(
                    dim + 1,
                    promote_int::<B>(tensor, n),
                    promote_int::<B>(indices, n),
                    promote_int::<B>(value, n),
                    update,
                ),
                n,
            ),
            None => BatchedPrimitive::unbatched(B::int_scatter(
                dim,
                tensor.primitive,
                indices.primitive,
                value.primitive,
                update,
            )),
        }
    }

    fn int_select(tensor: BatchedInt<B>, dim: usize, indices: BatchedInt<B>) -> BatchedInt<B> {
        match (tensor.batch_size, indices.batch_size) {
            (None, None) => {
                BatchedPrimitive::unbatched(B::int_select(tensor.primitive, dim, indices.primitive))
            }
            (Some(n), None) => BatchedPrimitive::batched(
                B::int_select(tensor.primitive, dim + 1, indices.primitive),
                n,
            ),
            (None, Some(n)) => BatchedPrimitive::batched(
                B::int_gather(dim + 1, promote_int::<B>(tensor, n), indices.primitive),
                n,
            ),
            (Some(a), Some(b)) => {
                assert_eq!(a, b);
                BatchedPrimitive::batched(
                    B::int_gather(dim + 1, tensor.primitive, indices.primitive),
                    a,
                )
            }
        }
    }

    fn int_select_assign(
        tensor: BatchedInt<B>,
        dim: usize,
        indices: BatchedInt<B>,
        value: BatchedInt<B>,
        update: burn_backend::tensor::IndexingUpdateOp,
    ) -> BatchedInt<B> {
        let batch_size = tensor
            .batch_size
            .or(indices.batch_size)
            .or(value.batch_size);
        match batch_size {
            Some(n) => BatchedPrimitive::batched(
                B::int_select_assign(
                    promote_int::<B>(tensor, n),
                    dim + 1,
                    promote_int::<B>(indices, n),
                    promote_int::<B>(value, n),
                    update,
                ),
                n,
            ),
            None => BatchedPrimitive::unbatched(B::int_select_assign(
                tensor.primitive,
                dim,
                indices.primitive,
                value.primitive,
                update,
            )),
        }
    }

    fn int_equal(lhs: BatchedInt<B>, rhs: BatchedInt<B>, out_dtype: BoolStore) -> BatchedBool<B> {
        let (lhs, rhs, batch_size) = unify_int::<B>(lhs, rhs);
        BatchedPrimitive {
            primitive: B::int_equal(lhs, rhs, out_dtype),
            batch_size,
        }
    }

    fn int_equal_elem(lhs: BatchedInt<B>, rhs: Scalar, out_dtype: BoolStore) -> BatchedBool<B> {
        BatchedPrimitive {
            primitive: B::int_equal_elem(lhs.primitive, rhs, out_dtype),
            batch_size: lhs.batch_size,
        }
    }

    fn int_greater(lhs: BatchedInt<B>, rhs: BatchedInt<B>, out_dtype: BoolStore) -> BatchedBool<B> {
        let (lhs, rhs, batch_size) = unify_int::<B>(lhs, rhs);
        BatchedPrimitive {
            primitive: B::int_greater(lhs, rhs, out_dtype),
            batch_size,
        }
    }

    fn int_greater_elem(lhs: BatchedInt<B>, rhs: Scalar, out_dtype: BoolStore) -> BatchedBool<B> {
        BatchedPrimitive {
            primitive: B::int_greater_elem(lhs.primitive, rhs, out_dtype),
            batch_size: lhs.batch_size,
        }
    }

    fn int_greater_equal(
        lhs: BatchedInt<B>,
        rhs: BatchedInt<B>,
        out_dtype: BoolStore,
    ) -> BatchedBool<B> {
        let (lhs, rhs, batch_size) = unify_int::<B>(lhs, rhs);
        BatchedPrimitive {
            primitive: B::int_greater_equal(lhs, rhs, out_dtype),
            batch_size,
        }
    }

    fn int_greater_equal_elem(
        lhs: BatchedInt<B>,
        rhs: Scalar,
        out_dtype: BoolStore,
    ) -> BatchedBool<B> {
        BatchedPrimitive {
            primitive: B::int_greater_equal_elem(lhs.primitive, rhs, out_dtype),
            batch_size: lhs.batch_size,
        }
    }

    fn int_lower(lhs: BatchedInt<B>, rhs: BatchedInt<B>, out_dtype: BoolStore) -> BatchedBool<B> {
        let (lhs, rhs, batch_size) = unify_int::<B>(lhs, rhs);
        BatchedPrimitive {
            primitive: B::int_lower(lhs, rhs, out_dtype),
            batch_size,
        }
    }

    fn int_lower_elem(lhs: BatchedInt<B>, rhs: Scalar, out_dtype: BoolStore) -> BatchedBool<B> {
        BatchedPrimitive {
            primitive: B::int_lower_elem(lhs.primitive, rhs, out_dtype),
            batch_size: lhs.batch_size,
        }
    }

    fn int_lower_equal(
        lhs: BatchedInt<B>,
        rhs: BatchedInt<B>,
        out_dtype: BoolStore,
    ) -> BatchedBool<B> {
        let (lhs, rhs, batch_size) = unify_int::<B>(lhs, rhs);
        BatchedPrimitive {
            primitive: B::int_lower_equal(lhs, rhs, out_dtype),
            batch_size,
        }
    }

    fn int_lower_equal_elem(
        lhs: BatchedInt<B>,
        rhs: Scalar,
        out_dtype: BoolStore,
    ) -> BatchedBool<B> {
        BatchedPrimitive {
            primitive: B::int_lower_equal_elem(lhs.primitive, rhs, out_dtype),
            batch_size: lhs.batch_size,
        }
    }

    int_binary!(int_add, int_add);
    int_binary_scalar!(int_add_scalar, int_add_scalar);
    int_binary!(int_sub, int_sub);
    int_binary_scalar!(int_sub_scalar, int_sub_scalar);
    int_binary!(int_mul, int_mul);
    int_binary_scalar!(int_mul_scalar, int_mul_scalar);
    int_binary!(int_div, int_div);
    int_binary_scalar!(int_div_scalar, int_div_scalar);
    int_binary!(int_remainder, int_remainder);
    int_binary_scalar!(int_remainder_scalar, int_remainder_scalar);
    int_binary!(int_matmul, int_matmul);

    fn int_sum(tensor: BatchedInt<B>) -> BatchedInt<B> {
        match tensor.batch_size {
            None => BatchedPrimitive::unbatched(B::int_sum(tensor.primitive)),
            Some(n) => {
                let numel = tensor.primitive.shape().num_elements() / n;
                let reshaped = B::int_reshape(tensor.primitive, Shape::from(alloc::vec![n, numel]));
                BatchedPrimitive::batched(B::int_sum_dim(reshaped, 1), n)
            }
        }
    }

    int_dim!(int_sum_dim, int_sum_dim);
    int_dim!(int_prod_dim, int_prod_dim);
    int_dim!(int_mean_dim, int_mean_dim);
    int_dim!(int_cumsum, int_cumsum);
    int_dim!(int_cumprod, int_cumprod);
    int_dim!(int_cummin, int_cummin);
    int_dim!(int_cummax, int_cummax);
    int_dim!(int_argmax, int_argmax);
    int_dim!(int_argmin, int_argmin);

    fn int_prod(tensor: BatchedInt<B>) -> BatchedInt<B> {
        match tensor.batch_size {
            None => BatchedPrimitive::unbatched(B::int_prod(tensor.primitive)),
            Some(n) => {
                let numel = tensor.primitive.shape().num_elements() / n;
                let reshaped = B::int_reshape(tensor.primitive, Shape::from(alloc::vec![n, numel]));
                BatchedPrimitive::batched(B::int_prod_dim(reshaped, 1), n)
            }
        }
    }

    int_unary!(int_abs, int_abs);

    async fn int_mask_select(tensor: BatchedInt<B>, mask: BatchedBool<B>) -> BatchedInt<B> {
        assert!(
            tensor.batch_size.is_none() && mask.batch_size.is_none(),
            "int_mask_select is not supported on batched (vmap) tensors; \
             close over masks outside the vmap scope or call int_mask_select per-sample"
        );
        BatchedPrimitive::unbatched(B::int_mask_select(tensor.primitive, mask.primitive).await)
    }

    fn int_swap_dims(tensor: BatchedInt<B>, dim1: usize, dim2: usize) -> BatchedInt<B> {
        BatchedPrimitive {
            primitive: B::int_swap_dims(
                tensor.primitive,
                batched_axis(tensor.batch_size, dim1),
                batched_axis(tensor.batch_size, dim2),
            ),
            batch_size: tensor.batch_size,
        }
    }

    fn int_permute(tensor: BatchedInt<B>, axes: &[usize]) -> BatchedInt<B> {
        let axes = match tensor.batch_size {
            Some(_) => {
                let mut axes = axes
                    .iter()
                    .map(|axis| axis + 1)
                    .collect::<alloc::vec::Vec<_>>();
                axes.insert(0, 0);
                axes
            }
            None => axes.to_vec(),
        };
        BatchedPrimitive {
            primitive: B::int_permute(tensor.primitive, &axes),
            batch_size: tensor.batch_size,
        }
    }

    fn int_flip(tensor: BatchedInt<B>, axes: &[usize]) -> BatchedInt<B> {
        let axes = match tensor.batch_size {
            Some(_) => axes
                .iter()
                .map(|axis| axis + 1)
                .collect::<alloc::vec::Vec<_>>(),
            None => axes.to_vec(),
        };
        BatchedPrimitive {
            primitive: B::int_flip(tensor.primitive, &axes),
            batch_size: tensor.batch_size,
        }
    }

    fn int_random(
        shape: Shape,
        distribution: Distribution,
        device: &B::Device,
        dtype: IntDType,
    ) -> BatchedInt<B> {
        BatchedPrimitive::unbatched(B::int_random(shape, distribution, device, dtype))
    }

    fn int_expand(tensor: BatchedInt<B>, shape: Shape) -> BatchedInt<B> {
        let shape = super::batched_shape(tensor.batch_size, &shape);
        BatchedPrimitive {
            primitive: B::int_expand(tensor.primitive, shape),
            batch_size: tensor.batch_size,
        }
    }

    fn int_cast(tensor: BatchedInt<B>, dtype: IntDType) -> BatchedInt<B> {
        BatchedPrimitive {
            primitive: B::int_cast(tensor.primitive, dtype),
            batch_size: tensor.batch_size,
        }
    }

    fn int_unfold(tensor: BatchedInt<B>, dim: usize, size: usize, step: usize) -> BatchedInt<B> {
        BatchedPrimitive {
            primitive: B::int_unfold(
                tensor.primitive,
                batched_axis(tensor.batch_size, dim),
                size,
                step,
            ),
            batch_size: tensor.batch_size,
        }
    }

    int_binary!(bitwise_and, bitwise_and);
    int_binary_scalar!(bitwise_and_scalar, bitwise_and_scalar);
    int_binary!(bitwise_or, bitwise_or);
    int_binary_scalar!(bitwise_or_scalar, bitwise_or_scalar);
    int_binary!(bitwise_xor, bitwise_xor);
    int_binary_scalar!(bitwise_xor_scalar, bitwise_xor_scalar);
    int_unary!(bitwise_not, bitwise_not);
    int_binary!(bitwise_left_shift, bitwise_left_shift);
    int_binary_scalar!(bitwise_left_shift_scalar, bitwise_left_shift_scalar);
    int_binary!(bitwise_right_shift, bitwise_right_shift);
    int_binary_scalar!(bitwise_right_shift_scalar, bitwise_right_shift_scalar);
}

macro_rules! bool_binary {
    ($name:ident, $backend:ident) => {
        fn $name(lhs: BatchedBool<B>, rhs: BatchedBool<B>) -> BatchedBool<B> {
            let (lhs, rhs, batch_size) = unify_bool::<B>(lhs, rhs);
            BatchedPrimitive {
                primitive: B::$backend(lhs, rhs),
                batch_size,
            }
        }
    };
}

macro_rules! bool_unary {
    ($name:ident, $backend:ident) => {
        fn $name(tensor: BatchedBool<B>) -> BatchedBool<B> {
            BatchedPrimitive {
                primitive: B::$backend(tensor.primitive),
                batch_size: tensor.batch_size,
            }
        }
    };
}

impl<B: Backend> BoolTensorOps<Vmap<B>> for Vmap<B> {
    fn bool_empty(shape: Shape, device: &B::Device, dtype: BoolStore) -> BatchedBool<B> {
        BatchedPrimitive::unbatched(B::bool_empty(shape, device, dtype))
    }

    fn bool_zeros(shape: Shape, device: &B::Device, dtype: BoolStore) -> BatchedBool<B> {
        BatchedPrimitive::unbatched(B::bool_zeros(shape, device, dtype))
    }

    fn bool_ones(shape: Shape, device: &B::Device, dtype: BoolStore) -> BatchedBool<B> {
        BatchedPrimitive::unbatched(B::bool_ones(shape, device, dtype))
    }

    fn bool_into_data(
        tensor: BatchedBool<B>,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send {
        B::bool_into_data(tensor.primitive)
    }

    fn bool_from_data(data: TensorData, device: &B::Device) -> BatchedBool<B> {
        BatchedPrimitive::unbatched(B::bool_from_data(data, device))
    }

    fn bool_into_int(tensor: BatchedBool<B>, dtype: IntDType) -> BatchedInt<B> {
        BatchedPrimitive {
            primitive: B::bool_into_int(tensor.primitive, dtype),
            batch_size: tensor.batch_size,
        }
    }

    fn bool_into_float(tensor: BatchedBool<B>, dtype: FloatDType) -> BatchedFloat<B> {
        BatchedPrimitive {
            primitive: B::bool_into_float(tensor.primitive, dtype),
            batch_size: tensor.batch_size,
        }
    }

    fn bool_to_device(tensor: BatchedBool<B>, device: &B::Device) -> BatchedBool<B> {
        BatchedPrimitive {
            primitive: B::bool_to_device(tensor.primitive, device),
            batch_size: tensor.batch_size,
        }
    }

    fn bool_reshape(tensor: BatchedBool<B>, shape: Shape) -> BatchedBool<B> {
        let shape = super::batched_shape(tensor.batch_size, &shape);
        BatchedPrimitive {
            primitive: B::bool_reshape(tensor.primitive, shape),
            batch_size: tensor.batch_size,
        }
    }

    fn bool_slice(tensor: BatchedBool<B>, slices: &[Slice]) -> BatchedBool<B> {
        let slices = prepend_full_batch_slice(tensor.batch_size, slices);
        BatchedPrimitive {
            primitive: B::bool_slice(tensor.primitive, &slices),
            batch_size: tensor.batch_size,
        }
    }

    fn bool_slice_assign(
        tensor: BatchedBool<B>,
        slices: &[Slice],
        value: BatchedBool<B>,
    ) -> BatchedBool<B> {
        let batch_size = tensor.batch_size.or(value.batch_size);
        let (tensor, value) = match batch_size {
            Some(n) => (promote_bool::<B>(tensor, n), promote_bool::<B>(value, n)),
            None => (tensor.primitive, value.primitive),
        };
        let slices = prepend_full_batch_slice(batch_size, slices);
        BatchedPrimitive {
            primitive: B::bool_slice_assign(tensor, &slices, value),
            batch_size,
        }
    }

    fn bool_mask_where(
        tensor: BatchedBool<B>,
        mask: BatchedBool<B>,
        value: BatchedBool<B>,
    ) -> BatchedBool<B> {
        let batch_size = tensor.batch_size.or(mask.batch_size).or(value.batch_size);
        match batch_size {
            Some(n) => BatchedPrimitive::batched(
                B::bool_mask_where(
                    promote_bool::<B>(tensor, n),
                    promote_bool::<B>(mask, n),
                    promote_bool::<B>(value, n),
                ),
                n,
            ),
            None => BatchedPrimitive::unbatched(B::bool_mask_where(
                tensor.primitive,
                mask.primitive,
                value.primitive,
            )),
        }
    }

    fn bool_mask_fill(
        tensor: BatchedBool<B>,
        mask: BatchedBool<B>,
        value: Scalar,
    ) -> BatchedBool<B> {
        let batch_size = tensor.batch_size.or(mask.batch_size);
        match batch_size {
            Some(n) => BatchedPrimitive::batched(
                B::bool_mask_fill(
                    promote_bool::<B>(tensor, n),
                    promote_bool::<B>(mask, n),
                    value,
                ),
                n,
            ),
            None => BatchedPrimitive::unbatched(B::bool_mask_fill(
                tensor.primitive,
                mask.primitive,
                value,
            )),
        }
    }

    fn bool_gather(dim: usize, tensor: BatchedBool<B>, indices: BatchedInt<B>) -> BatchedBool<B> {
        let batch_size = tensor.batch_size.or(indices.batch_size);
        match batch_size {
            Some(n) => BatchedPrimitive::batched(
                B::bool_gather(
                    dim + 1,
                    promote_bool::<B>(tensor, n),
                    promote_int::<B>(indices, n),
                ),
                n,
            ),
            None => BatchedPrimitive::unbatched(B::bool_gather(
                dim,
                tensor.primitive,
                indices.primitive,
            )),
        }
    }

    fn bool_scatter(
        dim: usize,
        tensor: BatchedBool<B>,
        indices: BatchedInt<B>,
        value: BatchedBool<B>,
        update: burn_backend::tensor::IndexingUpdateOp,
    ) -> BatchedBool<B> {
        let batch_size = tensor
            .batch_size
            .or(indices.batch_size)
            .or(value.batch_size);
        match batch_size {
            Some(n) => BatchedPrimitive::batched(
                B::bool_scatter(
                    dim + 1,
                    promote_bool::<B>(tensor, n),
                    promote_int::<B>(indices, n),
                    promote_bool::<B>(value, n),
                    update,
                ),
                n,
            ),
            None => BatchedPrimitive::unbatched(B::bool_scatter(
                dim,
                tensor.primitive,
                indices.primitive,
                value.primitive,
                update,
            )),
        }
    }

    fn bool_scatter_nd(
        tensor: BatchedBool<B>,
        indices: BatchedInt<B>,
        value: BatchedBool<B>,
        update: burn_backend::tensor::IndexingUpdateOp,
    ) -> BatchedBool<B> {
        let batch_size = tensor
            .batch_size
            .or(indices.batch_size)
            .or(value.batch_size);
        match batch_size {
            Some(n) => BatchedPrimitive::batched(
                B::bool_scatter_nd(
                    promote_bool::<B>(tensor, n),
                    promote_int::<B>(indices, n),
                    promote_bool::<B>(value, n),
                    update,
                ),
                n,
            ),
            None => BatchedPrimitive::unbatched(B::bool_scatter_nd(
                tensor.primitive,
                indices.primitive,
                value.primitive,
                update,
            )),
        }
    }

    fn bool_gather_nd(tensor: BatchedBool<B>, indices: BatchedInt<B>) -> BatchedBool<B> {
        let batch_size = tensor.batch_size.or(indices.batch_size);
        match batch_size {
            Some(n) => BatchedPrimitive::batched(
                B::bool_gather_nd(promote_bool::<B>(tensor, n), promote_int::<B>(indices, n)),
                n,
            ),
            None => {
                BatchedPrimitive::unbatched(B::bool_gather_nd(tensor.primitive, indices.primitive))
            }
        }
    }

    fn bool_select(tensor: BatchedBool<B>, dim: usize, indices: BatchedInt<B>) -> BatchedBool<B> {
        match (tensor.batch_size, indices.batch_size) {
            (None, None) => BatchedPrimitive::unbatched(B::bool_select(
                tensor.primitive,
                dim,
                indices.primitive,
            )),
            (Some(n), None) => BatchedPrimitive::batched(
                B::bool_select(tensor.primitive, dim + 1, indices.primitive),
                n,
            ),
            (None, Some(n)) => BatchedPrimitive::batched(
                B::bool_gather(dim + 1, promote_bool::<B>(tensor, n), indices.primitive),
                n,
            ),
            (Some(a), Some(b)) => {
                assert_eq!(a, b);
                BatchedPrimitive::batched(
                    B::bool_gather(dim + 1, tensor.primitive, indices.primitive),
                    a,
                )
            }
        }
    }

    fn bool_select_assign(
        tensor: BatchedBool<B>,
        dim: usize,
        indices: BatchedInt<B>,
        value: BatchedBool<B>,
        update: burn_backend::tensor::IndexingUpdateOp,
    ) -> BatchedBool<B> {
        let batch_size = tensor
            .batch_size
            .or(indices.batch_size)
            .or(value.batch_size);
        match batch_size {
            Some(n) => BatchedPrimitive::batched(
                B::bool_select_assign(
                    promote_bool::<B>(tensor, n),
                    dim + 1,
                    promote_int::<B>(indices, n),
                    promote_bool::<B>(value, n),
                    update,
                ),
                n,
            ),
            None => BatchedPrimitive::unbatched(B::bool_select_assign(
                tensor.primitive,
                dim,
                indices.primitive,
                value.primitive,
                update,
            )),
        }
    }

    bool_binary!(bool_equal, bool_equal);
    bool_binary!(bool_and, bool_and);
    bool_binary!(bool_or, bool_or);
    bool_unary!(bool_not, bool_not);

    fn bool_equal_elem(lhs: BatchedBool<B>, rhs: Scalar) -> BatchedBool<B> {
        BatchedPrimitive {
            primitive: B::bool_equal_elem(lhs.primitive, rhs),
            batch_size: lhs.batch_size,
        }
    }

    fn bool_swap_dims(tensor: BatchedBool<B>, dim1: usize, dim2: usize) -> BatchedBool<B> {
        BatchedPrimitive {
            primitive: B::bool_swap_dims(
                tensor.primitive,
                batched_axis(tensor.batch_size, dim1),
                batched_axis(tensor.batch_size, dim2),
            ),
            batch_size: tensor.batch_size,
        }
    }

    fn bool_permute(tensor: BatchedBool<B>, axes: &[usize]) -> BatchedBool<B> {
        let axes = match tensor.batch_size {
            Some(_) => {
                let mut axes = axes
                    .iter()
                    .map(|axis| axis + 1)
                    .collect::<alloc::vec::Vec<_>>();
                axes.insert(0, 0);
                axes
            }
            None => axes.to_vec(),
        };
        BatchedPrimitive {
            primitive: B::bool_permute(tensor.primitive, &axes),
            batch_size: tensor.batch_size,
        }
    }

    fn bool_flip(tensor: BatchedBool<B>, axes: &[usize]) -> BatchedBool<B> {
        let axes = match tensor.batch_size {
            Some(_) => axes
                .iter()
                .map(|axis| axis + 1)
                .collect::<alloc::vec::Vec<_>>(),
            None => axes.to_vec(),
        };
        BatchedPrimitive {
            primitive: B::bool_flip(tensor.primitive, &axes),
            batch_size: tensor.batch_size,
        }
    }

    fn bool_expand(tensor: BatchedBool<B>, shape: Shape) -> BatchedBool<B> {
        let shape = super::batched_shape(tensor.batch_size, &shape);
        BatchedPrimitive {
            primitive: B::bool_expand(tensor.primitive, shape),
            batch_size: tensor.batch_size,
        }
    }

    fn bool_unfold(tensor: BatchedBool<B>, dim: usize, size: usize, step: usize) -> BatchedBool<B> {
        BatchedPrimitive {
            primitive: B::bool_unfold(
                tensor.primitive,
                batched_axis(tensor.batch_size, dim),
                size,
                step,
            ),
            batch_size: tensor.batch_size,
        }
    }

    async fn bool_argwhere(tensor: BatchedBool<B>, out_dtype: IntDType) -> BatchedInt<B> {
        assert!(
            tensor.batch_size.is_none(),
            "bool_argwhere is not supported on batched (vmap) tensors; \
             close over masks outside the vmap scope or call bool_argwhere per-sample"
        );
        BatchedPrimitive::unbatched(B::bool_argwhere(tensor.primitive, out_dtype).await)
    }

    async fn bool_mask_select(tensor: BatchedBool<B>, mask: BatchedBool<B>) -> BatchedBool<B> {
        assert!(
            tensor.batch_size.is_none() && mask.batch_size.is_none(),
            "bool_mask_select is not supported on batched (vmap) tensors; \
             close over masks outside the vmap scope or call bool_mask_select per-sample"
        );
        BatchedPrimitive::unbatched(B::bool_mask_select(tensor.primitive, mask.primitive).await)
    }
}

impl<B: Backend> QTensorOps<Vmap<B>> for Vmap<B> {
    fn q_from_data(
        data: TensorData,
        device: &B::Device,
    ) -> burn_backend::tensor::QuantizedTensor<B> {
        B::q_from_data(data, device)
    }

    fn quantize(
        tensor: BatchedFloat<B>,
        scheme: &QuantScheme,
        qparams: QuantizationParametersPrimitive<Vmap<B>>,
    ) -> burn_backend::tensor::QuantizedTensor<B> {
        let qparams = QuantizationParametersPrimitive {
            scales: qparams.scales.primitive,
            global: qparams.global.map(|global| global.primitive),
        };
        B::quantize(tensor.primitive, scheme, qparams)
    }

    fn dequantize(
        tensor: burn_backend::tensor::QuantizedTensor<B>,
        dtype: FloatDType,
    ) -> BatchedFloat<B> {
        BatchedPrimitive::unbatched(B::dequantize(tensor, dtype))
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

impl<B: Backend> TransactionOps<Vmap<B>> for Vmap<B> {}

impl<B: Backend> DistributedOps<Vmap<B>> for Vmap<B> {}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use burn_backend::{TensorData, ops::ConvOptions};
    use burn_flex::{Flex, FlexDevice};

    use crate::{FuncTensor, Vmap, VmapAxis, vmap, vmap_with_axes};

    #[test]
    #[allow(clippy::type_complexity, clippy::identity_op)]
    fn vmap_conv2d_avg_pool2d_max_pool2d_attention() {
        let device = FlexDevice;
        // Physical shape [2, 1, 1, 4, 4] for batch size N=2 of logical 4D inputs [1, 1, 4, 4]
        let x_data = vec![1.0f32; 2 * 1 * 1 * 4 * 4];
        let x = FuncTensor::<Flex, 5>::from_data(TensorData::new(x_data, [2, 1, 1, 4, 4]), &device);

        // 1. conv2d with unbatched weight [1, 1, 3, 3]
        let weight = FuncTensor::<Flex, 4>::ones([1, 1, 3, 3], &device);
        let f_conv = |(x, w): &(FuncTensor<Vmap<Flex>, 5>, FuncTensor<Vmap<Flex>, 4>)| {
            x.clone()
                .conv2d(w.clone(), None, ConvOptions::new([1, 1], [0, 0], [1, 1], 1))
        };
        let out_conv: FuncTensor<Flex, 5> = vmap_with_axes::<Flex, _, _, _>(
            f_conv,
            vec![VmapAxis::Batched(0), VmapAxis::Unbatched],
        )(&(x.clone(), weight));
        assert_eq!(out_conv.shape().as_slice(), &[2, 1, 1, 2, 2]);
        let conv_vals = out_conv.into_data_sync().try_to_vec::<f32>().unwrap();
        assert_eq!(conv_vals, vec![9.0f32; 2 * 1 * 1 * 2 * 2]);

        // 2. avg_pool2d
        let f_avg =
            |x: &FuncTensor<Vmap<Flex>, 5>| x.clone().avg_pool2d([2, 2], [2, 2], [0, 0], false);
        let out_avg: FuncTensor<Flex, 5> = vmap::<Flex, _, _, _>(f_avg)(&x);
        assert_eq!(out_avg.shape().as_slice(), &[2, 1, 1, 2, 2]);
        let avg_vals = out_avg.into_data_sync().try_to_vec::<f32>().unwrap();
        assert_eq!(avg_vals, vec![1.0f32; 2 * 1 * 1 * 2 * 2]);

        // 3. max_pool2d
        let f_max =
            |x: &FuncTensor<Vmap<Flex>, 5>| x.clone().max_pool2d([2, 2], [2, 2], [0, 0], [1, 1]);
        let out_max: FuncTensor<Flex, 5> = vmap::<Flex, _, _, _>(f_max)(&x);
        assert_eq!(out_max.shape().as_slice(), &[2, 1, 1, 2, 2]);
        let max_vals = out_max.into_data_sync().try_to_vec::<f32>().unwrap();
        assert_eq!(max_vals, vec![1.0f32; 2 * 1 * 1 * 2 * 2]);

        // 4. attention: logical [1, 1, 2, 2], physical [2, 1, 1, 2, 2]
        let q = FuncTensor::<Flex, 5>::from_data(
            TensorData::new(vec![1.0f32; 2 * 1 * 1 * 2 * 2], [2, 1, 1, 2, 2]),
            &device,
        );
        let k = q.clone();
        let v = q.clone();
        let f_attn = |(q, k, v): &(
            FuncTensor<Vmap<Flex>, 5>,
            FuncTensor<Vmap<Flex>, 5>,
            FuncTensor<Vmap<Flex>, 5>,
        )| { q.clone().attention(k.clone(), v.clone()) };
        let out_attn: FuncTensor<Flex, 5> = vmap::<Flex, _, _, _>(f_attn)(&(q, k, v));
        assert_eq!(out_attn.shape().as_slice(), &[2, 1, 1, 2, 2]);
        let attn_vals = out_attn.into_data_sync().try_to_vec::<f32>().unwrap();
        assert!(attn_vals.iter().all(|&v| v.is_finite()));
    }
}
