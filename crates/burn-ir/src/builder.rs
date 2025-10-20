#![allow(missing_docs)]

use alloc::vec::Vec;
use burn_tensor::{
    DType, Distribution, Shape, Slice, calculate_matmul_output,
    ops::{
        conv::{
            calculate_conv_output_shape, calculate_conv_transpose_output_shape,
            calculate_pool_output_shape,
        },
        unfold::calculate_unfold_shape,
    },
    quantization::QuantScheme,
};

use crate::{ScalarIr, TensorId, TensorIr};

use super::operation::*;

#[derive(Debug)]
#[allow(missing_docs)]
pub enum IrError {
    DTypeMismatch,
}

fn output_dtype<'a, I>(inputs: I) -> Result<DType, IrError>
where
    I: IntoIterator<Item = &'a DType>,
{
    let mut iter = inputs.into_iter();
    let dtype = iter.next().unwrap();
    for d in iter {
        if d != dtype {
            return Err(IrError::DTypeMismatch);
        }
    }
    Ok(*dtype)
}

impl CreationOpIr {
    pub fn create(shape: Shape, dtype: DType, new_id: impl FnOnce() -> TensorId) -> Self {
        let out = TensorIr::uninit(new_id(), shape, dtype);

        CreationOpIr { out }
    }
}

impl InitOperationIr {
    pub fn create(shape: Shape, dtype: DType, new_id: impl FnOnce() -> TensorId) -> Self {
        let out = TensorIr::uninit(new_id(), shape, dtype);

        InitOperationIr { out }
    }
}

impl RandomOpIr {
    pub fn create(
        shape: Shape,
        dtype: DType,
        distribution: Distribution,
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        let out = TensorIr::uninit(new_id(), shape, dtype);

        RandomOpIr { out, distribution }
    }
}

impl FullOpIr {
    pub fn create(
        shape: Shape,
        dtype: DType,
        value: ScalarIr,
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        let out = TensorIr::uninit(new_id(), shape, dtype);

        FullOpIr { out, value }
    }
}

impl CastOpIr {
    pub fn create(input: TensorIr, dtype: DType, new_id: impl FnOnce() -> TensorId) -> Self {
        let out = TensorIr::uninit(new_id(), input.shape.clone(), dtype);
        CastOpIr { input, out }
    }
}

impl ShapeOpIr {
    pub fn expand(input: TensorIr, shape: Shape, new_id: impl FnOnce() -> TensorId) -> Self {
        let shape = input.shape.expand(shape).unwrap();
        Self::create(input, shape, new_id)
    }

    pub fn reshape(input: TensorIr, shape: Shape, new_id: impl FnOnce() -> TensorId) -> Self {
        let shape = input.shape.reshape(shape).unwrap();
        Self::create(input, shape, new_id)
    }

    fn create(input: TensorIr, shape: Shape, new_id: impl FnOnce() -> TensorId) -> Self {
        let out = TensorIr::uninit(new_id(), shape, input.dtype);
        ShapeOpIr { input, out }
    }
}

// "Lower" specific operations into a binary or unary op representation.
// Useful when collecting inputs and outputs and don't care about the other semantics.
impl From<MatmulOpIr> for BinaryOpIr {
    fn from(value: MatmulOpIr) -> Self {
        Self {
            lhs: value.lhs,
            rhs: value.rhs,
            out: value.out,
        }
    }
}

impl From<ReduceOpIr> for UnaryOpIr {
    fn from(value: ReduceOpIr) -> Self {
        Self {
            input: value.input,
            out: value.out,
        }
    }
}

/// Macro to implement `create` constructors for operations with a single output.
///
/// Supports shape and dtype validation.
macro_rules! impl_ir_create {
    (@create_fn $op:ident { $( $field:ident : $ty:ty ),* $(,)? } , $shape:expr, $dtype:expr) => {
        #[doc = "Create a new operation IR from the given inputs."]
        #[doc = "`new_id` should generate a unique `TensorId` for the uninitialized output tensor."]
        #[allow(clippy::too_many_arguments)]
        pub fn create($( $field : $ty ),*, new_id: impl FnOnce() -> crate::TensorId) -> $op {
            let shape = $shape;
            let dtype = $dtype;
            let out = TensorIr::uninit(new_id(), shape, dtype);
            $op { $( $field ),*, out }
        }
    };

    // Case: simple op, single `create`
    (
        $op:ident { $( $field:ident : $ty:ty ),* $(,)? },
        shape = $shape:expr,
        dtype = $dtype:expr
    ) => {
        impl $op {
            impl_ir_create!(@create_fn $op { $( $field : $ty ),* }, $shape, $dtype);
        }
    };

    // Case: op with one additional constructor that accepts an explicit output dtype
    (
        $op:ident { $( $field:ident : $ty:ty ),* $(,)? },
        shape = $shape:expr,
        dtype = $dtype:expr,
        $fn_name:ident ( $extra:ident : $extra_ty:ty )
    ) => {
        impl $op {
            impl_ir_create!(@create_fn $op { $( $field : $ty ),* }, $shape, $dtype);

            #[doc = "Create a new operation IR from the given inputs and the given output dtype."]
            #[allow(clippy::too_many_arguments)]
            pub fn $fn_name($( $field : $ty ),*, $extra: $extra_ty, new_id: impl FnOnce() -> crate::TensorId) -> Self {
                let shape = $shape;
                let _ = $dtype; // still validates dtype if needed
                let out = TensorIr::uninit(new_id(), shape, $extra);
                $op { $( $field ),*, out }
            }
        }
    };
}

impl_ir_create!(
    UnaryOpIr { input: TensorIr },
    shape = input.shape.clone(),
    dtype = input.dtype,
    // Additional constructor for unary comparisons
    create_comparison(bool_dtype: DType)
);

impl_ir_create!(
    BinaryOpIr {
        lhs: TensorIr,
        rhs: TensorIr
    },
    shape = lhs.shape.broadcast(&rhs.shape).unwrap(),
    dtype = output_dtype([&lhs.dtype, &rhs.dtype]).unwrap(),
    // Additional constructor for binary comparisons
    create_comparison(bool_dtype: DType)
);

impl_ir_create!(
    ScalarOpIr {
        lhs: TensorIr,
        rhs: ScalarIr
    },
    shape = lhs.shape.clone(),
    dtype = lhs.dtype,
    // Additional constructor for scalar comparisons
    create_comparison(bool_dtype: DType)
);

impl_ir_create!(
    MatmulOpIr {
        lhs: TensorIr,
        rhs: TensorIr
    },
    shape = calculate_matmul_output(&lhs.shape, &rhs.shape).unwrap(),
    dtype = output_dtype([&lhs.dtype, &rhs.dtype]).unwrap()
);

impl_ir_create!(
    SwapDimsOpIr {
        input: TensorIr,
        dim1: usize,
        dim2: usize
    },
    shape = input.shape.clone().swap(dim1, dim2).unwrap(),
    dtype = input.dtype
);

impl_ir_create!(
    PermuteOpIr { input: TensorIr, axes: Vec<usize> },
    shape = input.shape.clone().permute(&axes).unwrap(),
    dtype = input.dtype
);

impl_ir_create!(
    RepeatDimOpIr {
        tensor: TensorIr,
        dim: usize,
        times: usize
    },
    shape = tensor.shape.clone().repeat(dim, times).unwrap(),
    dtype = tensor.dtype
);

impl_ir_create!(
    FlipOpIr { input: TensorIr, axes: Vec<usize> },
    shape = input.shape.clone(), // TODO: check if axes are within the tensor dimensions
    dtype = input.dtype
);

impl_ir_create!(
    CatOpIr { tensors: Vec<TensorIr>, dim: usize },
    shape = Shape::cat(tensors.iter().map(|t| &t.shape), dim).unwrap(),
    dtype = output_dtype(tensors.iter().map(|t| &t.dtype)).unwrap()
);

impl_ir_create!(
    GatherOpIr {
        tensor: TensorIr,
        dim: usize,
        indices: TensorIr
    },
    shape = indices.shape.clone(), // TODO: check dims compat between tensor and indices
    dtype = tensor.dtype
);

impl_ir_create!(
    ScatterOpIr {
        tensor: TensorIr,
        dim: usize,
        indices: TensorIr,
        value: TensorIr
    },
    shape = tensor.shape.clone(), // TODO: check dims compat between tensor and indices
    dtype = output_dtype([&tensor.dtype, &value.dtype]).unwrap()
);

impl_ir_create!(
    ReduceOpIr { input: TensorIr },
    shape = [1].into(),
    dtype = input.dtype
);

impl_ir_create!(
    ReduceDimOpIr {
        input: TensorIr,
        axis: usize
    },
    shape = input.shape.clone().reduce(axis).unwrap(),
    dtype = input.dtype,
    // Additional constructor for argument reduction
    create_arg(ind_dtype: DType)
);

impl_ir_create!(
    DimOpIr {
        input: TensorIr,
        axis: usize
    },
    shape = input.shape.clone(), // TODO: check dims within rank
    dtype = input.dtype
);

impl_ir_create!(
    SelectOpIr {
        tensor: TensorIr,
        dim: usize,
        indices: TensorIr
    },
    // TODO: shape.select?
    shape = {
        let mut s = tensor.shape.clone();
        s[dim] = indices.shape[0];
        s
    },
    dtype = tensor.dtype
);

impl_ir_create!(
    SelectAssignOpIr {
        tensor: TensorIr,
        dim: usize,
        indices: TensorIr,
        value: TensorIr
    },
    // TODO: check value and indices shape match for dim
    shape = tensor.shape.clone(),
    dtype = output_dtype([&tensor.dtype, &value.dtype]).unwrap()
);

impl_ir_create!(
    SliceOpIr {
        tensor: TensorIr,
        ranges: Vec<Slice>,
    },
    shape = tensor.shape.clone().slice(&ranges).unwrap(),
    dtype = tensor.dtype
);

impl_ir_create!(
    SliceAssignOpIr {
        tensor: TensorIr,
        ranges: Vec<Slice>,
        value: TensorIr
    },
    // TODO: check slice and value number of elements match
    shape = tensor.shape.clone(),
    dtype = output_dtype([&tensor.dtype, &value.dtype]).unwrap()
);

impl_ir_create!(
    MaskWhereOpIr {
        tensor: TensorIr,
        mask: TensorIr,
        value: TensorIr
    },
    shape = Shape::broadcast_many([&tensor.shape, &mask.shape, &value.shape]).unwrap(),
    dtype = output_dtype([&tensor.dtype, &value.dtype]).unwrap()
);

impl_ir_create!(
    MaskFillOpIr {
        tensor: TensorIr,
        mask: TensorIr,
        value: ScalarIr
    },
    shape = tensor.shape.broadcast(&mask.shape).unwrap(),
    dtype = tensor.dtype
);

impl_ir_create!(
    ClampOpIr {
        tensor: TensorIr,
        min: ScalarIr,
        max: ScalarIr
    },
    shape = tensor.shape.clone(),
    dtype = tensor.dtype
);

impl_ir_create!(
    AvgPool1dOpIr {
        x: TensorIr,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool
    },
    shape =
        calculate_pool_output_shape(&x.shape, &[kernel_size], &[stride], &[padding], &[1]).unwrap(),
    dtype = x.dtype
);

impl_ir_create!(
    AvgPool1dBackwardOpIr {
        x: TensorIr,
        grad: TensorIr,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool
    },
    shape = x.shape.clone(),
    dtype = x.dtype
);

impl_ir_create!(
    AvgPool2dOpIr {
        x: TensorIr,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool
    },
    shape =
        calculate_pool_output_shape(&x.shape, &kernel_size, &stride, &padding, &[1, 1]).unwrap(),
    dtype = x.dtype
);

impl_ir_create!(
    AvgPool2dBackwardOpIr {
        x: TensorIr,
        grad: TensorIr,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool
    },
    shape = x.shape.clone(),
    dtype = x.dtype
);

impl_ir_create!(
    MaxPool1dOpIr {
        x: TensorIr,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    },
    shape =
        calculate_pool_output_shape(&x.shape, &[kernel_size], &[stride], &[padding], &[dilation])
            .unwrap(),
    dtype = x.dtype
);

impl_ir_create!(
    MaxPool2dOpIr {
        x: TensorIr,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    },
    shape =
        calculate_pool_output_shape(&x.shape, &kernel_size, &stride, &padding, &dilation).unwrap(),
    dtype = x.dtype
);

impl_ir_create!(
    MaxPool1dWithIndicesBackwardOpIr {
        x: TensorIr,
        grad: TensorIr,
        indices: TensorIr,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    },
    shape = x.shape.clone(),
    dtype = x.dtype
);

impl_ir_create!(
    MaxPool2dWithIndicesBackwardOpIr {
        x: TensorIr,
        grad: TensorIr,
        indices: TensorIr,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    },
    shape = x.shape.clone(),
    dtype = x.dtype
);

impl_ir_create!(
    AdaptiveAvgPool1dOpIr {
        x: TensorIr,
        output_size: usize
    },
    shape = Shape::new([x.shape[0], x.shape[1], output_size]),
    dtype = x.dtype
);

impl_ir_create!(
    AdaptiveAvgPool2dOpIr {
        x: TensorIr,
        output_size: [usize; 2]
    },
    shape = Shape::new([x.shape[0], x.shape[1], output_size[0], output_size[1]]),
    dtype = x.dtype
);

impl_ir_create!(
    AdaptiveAvgPool1dBackwardOpIr {
        x: TensorIr,
        grad: TensorIr,
    },
    shape = x.shape.clone(),
    dtype = x.dtype
);

impl_ir_create!(
    AdaptiveAvgPool2dBackwardOpIr {
        x: TensorIr,
        grad: TensorIr,
    },
    shape = x.shape.clone(),
    dtype = x.dtype
);

impl_ir_create!(
    InterpolateOpIr {
        x: TensorIr,
        output_size: [usize; 2],
        options: InterpolateOptionsIr
    },
    shape = Shape::new([x.shape[0], x.shape[1], output_size[0], output_size[1]]),
    dtype = x.dtype
);

impl_ir_create!(
    InterpolateBackwardOpIr {
        x: TensorIr,
        grad: TensorIr,
        output_size: [usize; 2],
        options: InterpolateOptionsIr
    },
    shape = x.shape.clone(),
    dtype = x.dtype
);

impl_ir_create!(
    Conv1dOpIr {
        x: TensorIr,
        weight: TensorIr,
        bias: Option<TensorIr>,
        options: Conv1dOptionsIr
    },
    shape = calculate_conv_output_shape(
            &x.shape,
            &weight.shape,
            &options.stride,
            &options.padding,
            &options.dilation,
        )
        .unwrap(),
    dtype = output_dtype(
            [
                Some(&x.dtype),
                Some(&weight.dtype),
                bias.as_ref().map(|b| &b.dtype),
            ]
            .iter()
            .filter_map(|&d| d),
        )
        .unwrap()
);

impl_ir_create!(
    Conv2dOpIr {
        x: TensorIr,
        weight: TensorIr,
        bias: Option<TensorIr>,
        options: Conv2dOptionsIr
    },
    shape = calculate_conv_output_shape(
            &x.shape,
            &weight.shape,
            &options.stride,
            &options.padding,
            &options.dilation,
        )
        .unwrap(),
    dtype = output_dtype(
            [
                Some(&x.dtype),
                Some(&weight.dtype),
                bias.as_ref().map(|b| &b.dtype),
            ]
            .iter()
            .filter_map(|&d| d),
        )
        .unwrap()
);

impl_ir_create!(
    Conv3dOpIr {
        x: TensorIr,
        weight: TensorIr,
        bias: Option<TensorIr>,
        options: Conv3dOptionsIr
    },
    shape = calculate_conv_output_shape(
            &x.shape,
            &weight.shape,
            &options.stride,
            &options.padding,
            &options.dilation,
        )
        .unwrap(),
    dtype = output_dtype(
            [
                Some(&x.dtype),
                Some(&weight.dtype),
                bias.as_ref().map(|b| &b.dtype),
            ]
            .iter()
            .filter_map(|&d| d),
        )
        .unwrap()
);

impl_ir_create!(
    DeformConv2dOpIr {
        x: TensorIr,
        offset: TensorIr,
        weight: TensorIr,
        mask: Option<TensorIr>,
        bias: Option<TensorIr>,
        options: DeformableConv2dOptionsIr
    },
    shape = calculate_conv_output_shape(
            &x.shape,
            &weight.shape,
            &options.stride,
            &options.padding,
            &options.dilation,
        )
        .unwrap(),
    dtype = output_dtype(
            [
                Some(&x.dtype),
                Some(&offset.dtype),
                Some(&weight.dtype),
                mask.as_ref().map(|m| &m.dtype),
                bias.as_ref().map(|b| &b.dtype),
            ]
            .iter()
            .filter_map(|&d| d),
        )
        .unwrap()
);

impl_ir_create!(
    ConvTranspose1dOpIr {
        x: TensorIr,
        weight: TensorIr,
        bias: Option<TensorIr>,
        options: ConvTranspose1dOptionsIr
    },
    shape = calculate_conv_transpose_output_shape(
            &x.shape,
            &weight.shape,
            &options.stride,
            &options.padding,
            &options.padding_out,
            &options.dilation,
            options.groups,
        )
        .unwrap(),
    dtype = output_dtype(
            [
                Some(&x.dtype),
                Some(&weight.dtype),
                bias.as_ref().map(|b| &b.dtype),
            ]
            .iter()
            .filter_map(|&d| d),
        )
        .unwrap()
);

impl_ir_create!(
    ConvTranspose2dOpIr {
        x: TensorIr,
        weight: TensorIr,
        bias: Option<TensorIr>,
        options: ConvTranspose2dOptionsIr
    },
    shape = calculate_conv_transpose_output_shape(
            &x.shape,
            &weight.shape,
            &options.stride,
            &options.padding,
            &options.padding_out,
            &options.dilation,
            options.groups,
        )
        .unwrap(),
    dtype = output_dtype(
            [
                Some(&x.dtype),
                Some(&weight.dtype),
                bias.as_ref().map(|b| &b.dtype),
            ]
            .iter()
            .filter_map(|&d| d),
        )
        .unwrap()
);

impl_ir_create!(
    ConvTranspose3dOpIr {
        x: TensorIr,
        weight: TensorIr,
        bias: Option<TensorIr>,
        options: ConvTranspose3dOptionsIr
    },
    shape = calculate_conv_transpose_output_shape(
            &x.shape,
            &weight.shape,
            &options.stride,
            &options.padding,
            &options.padding_out,
            &options.dilation,
            options.groups,
        )
        .unwrap(),
    dtype = output_dtype(
            [
                Some(&x.dtype),
                Some(&weight.dtype),
                bias.as_ref().map(|b| &b.dtype),
            ]
            .iter()
            .filter_map(|&d| d),
        )
        .unwrap()
);

impl_ir_create!(
    UnfoldOpIr {
        input: TensorIr,
        dim: usize,
        size: usize,
        step: usize
    },
    shape = calculate_unfold_shape(input.shape.clone(), dim, size, step),
    dtype = input.dtype
);

impl_ir_create!(
    CrossOpIr {
        lhs: TensorIr,
        rhs: TensorIr,
        dim: usize
    },
    shape = lhs.shape.broadcast(&rhs.shape).unwrap(),
    dtype = output_dtype([&lhs.dtype, &rhs.dtype]).unwrap()
);

impl_ir_create!(
    QuantizeOpIr {
        tensor: TensorIr,
        qparams: QuantizationParametersIr,
        scheme: QuantScheme
    },
    shape = tensor.shape.clone(),
    dtype = DType::QFloat(scheme)
);

impl DequantizeOpIr {
    pub fn create(input: TensorIr, dtype: DType, new_id: impl FnOnce() -> TensorId) -> Self {
        let out = TensorIr::uninit(new_id(), input.shape.clone(), dtype);

        DequantizeOpIr { input, out }
    }
}

// Operations with multiple outputs

impl ReduceDimWithIndicesOpIr {
    pub fn create(
        tensor: TensorIr,
        dim: usize,
        dtype_indices: DType,
        mut new_id: impl FnMut() -> TensorId,
    ) -> Self {
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = TensorIr::uninit(new_id(), shape.clone(), tensor.dtype);
        let out_indices = TensorIr::uninit(new_id(), shape.clone(), dtype_indices);

        ReduceDimWithIndicesOpIr {
            tensor,
            dim,
            out,
            out_indices,
        }
    }
}

impl DeformConv2dBackwardOpIr {
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        x: TensorIr,
        offset: TensorIr,
        weight: TensorIr,
        mask: Option<TensorIr>,
        bias: Option<TensorIr>,
        out_grad: TensorIr,
        options: DeformableConv2dOptionsIr,
        mut new_id: impl FnMut() -> TensorId,
    ) -> Self {
        let dtype = output_dtype(
            [
                Some(&x.dtype),
                Some(&weight.dtype),
                mask.as_ref().map(|m| &m.dtype),
                bias.as_ref().map(|b| &b.dtype),
            ]
            .iter()
            .filter_map(|&d| d),
        )
        .unwrap();

        let input_grad = TensorIr::uninit(new_id(), x.shape.clone(), dtype);
        let offset_grad = TensorIr::uninit(new_id(), offset.shape.clone(), dtype);
        let weight_grad = TensorIr::uninit(new_id(), weight.shape.clone(), dtype);
        let mask_grad = mask
            .as_ref()
            .map(|t| TensorIr::uninit(new_id(), t.shape.clone(), dtype));
        let bias_grad = bias
            .as_ref()
            .map(|t| TensorIr::uninit(new_id(), t.shape.clone(), dtype));

        DeformConv2dBackwardOpIr {
            x,
            offset,
            weight,
            mask,
            bias,
            out_grad,
            options,
            input_grad,
            offset_grad,
            weight_grad,
            mask_grad,
            bias_grad,
        }
    }
}

impl MaxPool1dWithIndicesOpIr {
    pub fn create(
        x: TensorIr,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        dtype_indices: DType,
        mut new_id: impl FnMut() -> TensorId,
    ) -> Self {
        let shape = calculate_pool_output_shape(
            &x.shape,
            &[kernel_size],
            &[stride],
            &[padding],
            &[dilation],
        )
        .unwrap();
        let out = TensorIr::uninit(new_id(), shape.clone(), x.dtype);
        let out_indices = TensorIr::uninit(new_id(), shape, dtype_indices);

        MaxPool1dWithIndicesOpIr {
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            out,
            out_indices,
        }
    }
}

impl MaxPool2dWithIndicesOpIr {
    pub fn create(
        x: TensorIr,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        dtype_indices: DType,
        mut new_id: impl FnMut() -> TensorId,
    ) -> Self {
        let shape =
            calculate_pool_output_shape(&x.shape, &kernel_size, &stride, &padding, &dilation)
                .unwrap();
        let out = TensorIr::uninit(new_id(), shape.clone(), x.dtype);
        let out_indices = TensorIr::uninit(new_id(), shape, dtype_indices);

        MaxPool2dWithIndicesOpIr {
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            out,
            out_indices,
        }
    }
}
