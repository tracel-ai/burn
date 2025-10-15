use burn_tensor::{
    DType, Distribution, Element, Shape, Slice, calculate_matmul_output,
    ops::{
        ConvOptions, ConvTransposeOptions, DeformConvOptions, InterpolateOptions,
        conv::{
            calculate_conv_output_shape, calculate_conv_transpose_output_shape,
            calculate_pool_output_shape,
        },
        unfold::calculate_unfold_shape,
    },
    quantization::QuantScheme,
};

use crate::{ScalarIr, TensorId, TensorIr, TensorStatus};

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
    pub fn create<E: Element>(
        shape: Shape,
        dtype: DType,
        value: E,
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        let value = ScalarIr::with_dtype(value, &dtype);
        let out = TensorIr {
            status: TensorStatus::NotInit,
            shape,
            id: new_id(),
            dtype,
        };
        FullOpIr { out, value }
    }
}

impl UnaryOpIr {
    pub fn create(input: TensorIr, new_id: impl FnOnce() -> TensorId) -> Self {
        let dtype = input.dtype;
        Self::create_with_dtype(input, dtype, new_id)
    }

    pub fn create_comparison(
        input: TensorIr,
        bool_dtype: DType,
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        Self::create_with_dtype(input, bool_dtype, new_id)
    }

    fn create_with_dtype(input: TensorIr, dtype: DType, new_id: impl FnOnce() -> TensorId) -> Self {
        let out = TensorIr::uninit(new_id(), input.shape.clone(), dtype);
        UnaryOpIr { input, out }
    }
}

impl BinaryOpIr {
    pub fn create(lhs: TensorIr, rhs: TensorIr, new_id: impl FnOnce() -> TensorId) -> Self {
        let dtype = lhs.dtype;
        Self::create_with_dtype(lhs, rhs, dtype, new_id)
    }

    pub fn create_comparison(
        lhs: TensorIr,
        rhs: TensorIr,
        bool_dtype: DType,
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        Self::create_with_dtype(lhs, rhs, bool_dtype, new_id)
    }

    fn create_with_dtype(
        lhs: TensorIr,
        rhs: TensorIr,
        dtype: DType,
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        let shape = lhs.shape.broadcast(&rhs.shape).unwrap();
        let _ = output_dtype([&lhs.dtype, &rhs.dtype]).unwrap();
        let out = TensorIr::uninit(new_id(), shape, dtype);
        BinaryOpIr { lhs, rhs, out }
    }
}

impl ScalarOpIr {
    pub fn create<E: Element>(lhs: TensorIr, rhs: E, new_id: impl FnOnce() -> TensorId) -> Self {
        let dtype = lhs.dtype;
        Self::create_with_dtype(lhs, rhs, dtype, new_id)
    }

    pub fn create_comparison<E: Element>(
        lhs: TensorIr,
        rhs: E,
        bool_dtype: DType,
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        Self::create_with_dtype(lhs, rhs, bool_dtype, new_id)
    }

    fn create_with_dtype<E: Element>(
        lhs: TensorIr,
        rhs: E,
        dtype: DType,
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let out = TensorIr::uninit(new_id(), lhs.shape.clone(), dtype);

        ScalarOpIr { lhs, rhs, out }
    }
}

impl MatmulOpIr {
    pub fn create(lhs: TensorIr, rhs: TensorIr, new_id: impl FnOnce() -> TensorId) -> Self {
        let shape = calculate_matmul_output(&lhs.shape, &rhs.shape).unwrap();
        let dtype = output_dtype([&lhs.dtype, &rhs.dtype]).unwrap();
        let out = TensorIr::uninit(new_id(), shape, dtype);
        MatmulOpIr { lhs, rhs, out }
    }
}

impl CastOpIr {
    pub fn create(input: TensorIr, dtype: DType, new_id: impl FnOnce() -> TensorId) -> Self {
        let out = TensorIr::uninit(new_id(), input.shape.clone(), dtype);
        CastOpIr { input, out }
    }
}

impl SwapDimsOpIr {
    pub fn create(
        input: TensorIr,
        dim1: usize,
        dim2: usize,
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        let shape = input.shape.clone().swap(dim1, dim2).unwrap();
        let out = TensorIr::uninit(new_id(), shape, input.dtype);

        SwapDimsOpIr {
            input,
            out,
            dim1,
            dim2,
        }
    }
}

impl PermuteOpIr {
    pub fn create(input: TensorIr, axes: Vec<usize>, new_id: impl FnOnce() -> TensorId) -> Self {
        let shape = input.shape.clone().permute(&axes).unwrap();
        let out = TensorIr::uninit(new_id(), shape, input.dtype);
        println!("permute out: {out:?}");

        PermuteOpIr { input, out, axes }
    }
}

impl ShapeOpIr {
    pub fn expand(input: TensorIr, shape: Shape, new_id: impl FnOnce() -> TensorId) -> Self {
        let shape = input.shape.expand(shape).unwrap();
        Self::new(input, shape, new_id)
    }

    pub fn reshape(input: TensorIr, shape: Shape, new_id: impl FnOnce() -> TensorId) -> Self {
        let shape = input.shape.reshape(shape).unwrap();
        Self::new(input, shape, new_id)
    }

    fn new(input: TensorIr, shape: Shape, new_id: impl FnOnce() -> TensorId) -> Self {
        let out = TensorIr::uninit(new_id(), shape, input.dtype);
        ShapeOpIr { input, out }
    }
}

impl FlipOpIr {
    pub fn create(input: TensorIr, axes: Vec<usize>, new_id: impl FnOnce() -> TensorId) -> Self {
        let out = TensorIr::uninit(new_id(), input.shape.clone(), input.dtype);

        FlipOpIr { input, out, axes }
    }
}

impl ReduceOpIr {
    pub fn create(input: TensorIr, new_id: impl FnOnce() -> TensorId) -> Self {
        let out = TensorIr::uninit(new_id(), [1].into(), input.dtype);
        ReduceOpIr { input, out }
    }
}

impl ReduceDimOpIr {
    pub fn create(input: TensorIr, axis: usize, new_id: impl FnOnce() -> TensorId) -> Self {
        let dtype = input.dtype;
        Self::create_with_dtype(input, axis, dtype, new_id)
    }

    pub fn create_with_dtype(
        input: TensorIr,
        axis: usize,
        dtype: DType,
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        let mut shape = input.shape.clone();
        shape[axis] = 1;
        let out = TensorIr::uninit(new_id(), shape, dtype);

        ReduceDimOpIr { input, out, axis }
    }
}

impl DimOpIr {
    pub fn create(input: TensorIr, axis: usize, new_id: impl FnOnce() -> TensorId) -> Self {
        let out = TensorIr::uninit(new_id(), input.shape.clone(), input.dtype);

        DimOpIr { input, out, axis }
    }
}

impl GatherOpIr {
    pub fn create(
        tensor: TensorIr,
        dim: usize,
        indices: TensorIr,
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        let out = TensorIr::uninit(new_id(), indices.shape.clone(), tensor.dtype);

        GatherOpIr {
            tensor,
            dim,
            indices,
            out,
        }
    }
}

impl ScatterOpIr {
    pub fn create(
        tensor: TensorIr,
        dim: usize,
        indices: TensorIr,
        value: TensorIr,
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        let dtype = output_dtype([&tensor.dtype, &value.dtype]).unwrap();
        let out = TensorIr::uninit(new_id(), tensor.shape.clone(), dtype);

        ScatterOpIr {
            tensor,
            dim,
            indices,
            value,
            out,
        }
    }
}

impl SelectOpIr {
    pub fn create(
        tensor: TensorIr,
        dim: usize,
        indices: TensorIr,
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        let mut shape = tensor.shape.clone();
        shape[dim] = indices.shape[0];
        let out = TensorIr::uninit(new_id(), shape, tensor.dtype);

        SelectOpIr {
            tensor,
            dim,
            indices,
            out,
        }
    }
}

impl SelectAssignOpIr {
    pub fn create(
        tensor: TensorIr,
        dim: usize,
        indices: TensorIr,
        value: TensorIr,
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        let dtype = output_dtype([&tensor.dtype, &value.dtype]).unwrap();
        let out = TensorIr::uninit(new_id(), tensor.shape.clone(), dtype);

        SelectAssignOpIr {
            tensor,
            dim,
            indices,
            value,
            out,
        }
    }
}

impl SliceOpIr {
    pub fn create(tensor: TensorIr, slices: Vec<Slice>, new_id: impl FnOnce() -> TensorId) -> Self {
        let shape = tensor.shape.clone().slice(&slices).unwrap();
        let out = TensorIr::uninit(new_id(), shape, tensor.dtype);

        SliceOpIr {
            tensor,
            ranges: slices,
            out,
        }
    }
}

impl SliceAssignOpIr {
    pub fn create(
        tensor: TensorIr,
        slices: Vec<Slice>,
        value: TensorIr,
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        let dtype = output_dtype([&tensor.dtype, &value.dtype]).unwrap();
        let out = TensorIr::uninit(new_id(), tensor.shape.clone(), dtype);

        SliceAssignOpIr {
            tensor,
            ranges: slices,
            value,
            out,
        }
    }
}

impl MaskWhereOpIr {
    pub fn create(
        tensor: TensorIr,
        mask: TensorIr,
        value: TensorIr,
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        let dtype = output_dtype([&tensor.dtype, &value.dtype]).unwrap();
        let shape = Shape::broadcast_many([&tensor.shape, &mask.shape, &value.shape]).unwrap();
        let out = TensorIr::uninit(new_id(), shape, dtype);

        MaskWhereOpIr {
            tensor,
            mask,
            value,
            out,
        }
    }
}

impl MaskFillOpIr {
    pub fn create<E: Element>(
        tensor: TensorIr,
        mask: TensorIr,
        value: E,
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        let shape = tensor.shape.broadcast(&mask.shape).unwrap();
        let dtype = tensor.dtype;
        let value = ScalarIr::with_dtype(value, &dtype);
        let out = TensorIr::uninit(new_id(), shape, dtype);

        MaskFillOpIr {
            tensor,
            mask,
            value,
            out,
        }
    }
}

impl ClampOpIr {
    pub fn create<E: Element>(
        tensor: TensorIr,
        min: E,
        max: E,
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        let dtype = tensor.dtype;
        let min = ScalarIr::with_dtype(min, &dtype);
        let max = ScalarIr::with_dtype(max, &dtype);
        let out = TensorIr::uninit(new_id(), tensor.shape.clone(), dtype);

        ClampOpIr {
            tensor,
            min,
            max,
            out,
        }
    }
}

impl RepeatDimOpIr {
    pub fn create(
        tensor: TensorIr,
        dim: usize,
        times: usize,
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        let shape = tensor.shape.clone().repeat(dim, times);
        let out = TensorIr::uninit(new_id(), shape, tensor.dtype);

        RepeatDimOpIr {
            tensor,
            dim,
            times,
            out,
        }
    }
}

impl CatOpIr {
    pub fn create(tensors: Vec<TensorIr>, dim: usize, new_id: impl FnOnce() -> TensorId) -> Self {
        let dtype = output_dtype(tensors.iter().map(|t| &t.dtype)).unwrap();
        let shape = Shape::cat(tensors.iter().map(|t| &t.shape), dim).unwrap();
        let out = TensorIr::uninit(new_id(), shape, dtype);

        CatOpIr { tensors, dim, out }
    }
}

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

impl Conv1dOpIr {
    pub fn create(
        x: TensorIr,
        weight: TensorIr,
        bias: Option<TensorIr>,
        options: ConvOptions<1>,
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        let shape = calculate_conv_output_shape(
            &x.shape,
            &weight.shape,
            &options.stride,
            &options.padding,
            &options.dilation,
        )
        .unwrap();
        let dtype = output_dtype([
            &x.dtype,
            &weight.dtype,
            bias.as_ref().map(|b| &b.dtype).unwrap_or(&x.dtype),
        ])
        .unwrap();
        let out = TensorIr::uninit(new_id(), shape, dtype);

        Conv1dOpIr {
            x,
            weight,
            bias,
            options: options.into(),
            out,
        }
    }
}

impl Conv2dOpIr {
    pub fn create(
        x: TensorIr,
        weight: TensorIr,
        bias: Option<TensorIr>,
        options: ConvOptions<2>,
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        let shape = calculate_conv_output_shape(
            &x.shape,
            &weight.shape,
            &options.stride,
            &options.padding,
            &options.dilation,
        )
        .unwrap();
        let dtype = output_dtype([
            &x.dtype,
            &weight.dtype,
            bias.as_ref().map(|b| &b.dtype).unwrap_or(&x.dtype),
        ])
        .unwrap();
        let out = TensorIr::uninit(new_id(), shape, dtype);

        Conv2dOpIr {
            x,
            weight,
            bias,
            options: options.into(),
            out,
        }
    }
}

impl Conv3dOpIr {
    pub fn create(
        x: TensorIr,
        weight: TensorIr,
        bias: Option<TensorIr>,
        options: ConvOptions<3>,
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        let shape = calculate_conv_output_shape(
            &x.shape,
            &weight.shape,
            &options.stride,
            &options.padding,
            &options.dilation,
        )
        .unwrap();
        let dtype = output_dtype([
            &x.dtype,
            &weight.dtype,
            bias.as_ref().map(|b| &b.dtype).unwrap_or(&x.dtype),
        ])
        .unwrap();
        let out = TensorIr::uninit(new_id(), shape, dtype);

        Conv3dOpIr {
            x,
            weight,
            bias,
            options: options.into(),
            out,
        }
    }
}

impl DeformConv2dOpIr {
    pub fn create(
        x: TensorIr,
        offset: TensorIr,
        weight: TensorIr,
        mask: Option<TensorIr>,
        bias: Option<TensorIr>,
        options: DeformConvOptions<2>,
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        let shape = calculate_conv_output_shape(
            &x.shape,
            &weight.shape,
            &options.stride,
            &options.padding,
            &options.dilation,
        )
        .unwrap();
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
        let out = TensorIr::uninit(new_id(), shape, dtype);

        DeformConv2dOpIr {
            x,
            offset,
            weight,
            mask,
            bias,
            options: options.into(),
            out,
        }
    }
}

impl DeformConv2dBackwardOpIr {
    pub fn create(
        x: TensorIr,
        offset: TensorIr,
        weight: TensorIr,
        mask: Option<TensorIr>,
        bias: Option<TensorIr>,
        out_grad: TensorIr,
        options: DeformConvOptions<2>,
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
            options: options.into(),
            input_grad,
            offset_grad,
            weight_grad,
            mask_grad,
            bias_grad,
        }
    }
}

impl ConvTranspose1dOpIr {
    pub fn create(
        x: TensorIr,
        weight: TensorIr,
        bias: Option<TensorIr>,
        options: ConvTransposeOptions<1>,
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        let shape = calculate_conv_transpose_output_shape(
            &x.shape,
            &weight.shape,
            &options.stride,
            &options.padding,
            &options.padding_out,
            &options.dilation,
            options.groups,
        )
        .unwrap();
        let dtype = output_dtype(
            [
                Some(&x.dtype),
                Some(&weight.dtype),
                bias.as_ref().map(|b| &b.dtype),
            ]
            .iter()
            .filter_map(|&d| d),
        )
        .unwrap();
        let out = TensorIr::uninit(new_id(), shape, dtype);

        ConvTranspose1dOpIr {
            x,
            weight,
            bias,
            options: options.into(),
            out,
        }
    }
}

impl ConvTranspose2dOpIr {
    pub fn create(
        x: TensorIr,
        weight: TensorIr,
        bias: Option<TensorIr>,
        options: ConvTransposeOptions<2>,
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        let shape = calculate_conv_transpose_output_shape(
            &x.shape,
            &weight.shape,
            &options.stride,
            &options.padding,
            &options.padding_out,
            &options.dilation,
            options.groups,
        )
        .unwrap();
        let dtype = output_dtype(
            [
                Some(&x.dtype),
                Some(&weight.dtype),
                bias.as_ref().map(|b| &b.dtype),
            ]
            .iter()
            .filter_map(|&d| d),
        )
        .unwrap();
        let out = TensorIr::uninit(new_id(), shape, dtype);

        ConvTranspose2dOpIr {
            x,
            weight,
            bias,
            options: options.into(),
            out,
        }
    }
}

impl ConvTranspose3dOpIr {
    pub fn create(
        x: TensorIr,
        weight: TensorIr,
        bias: Option<TensorIr>,
        options: ConvTransposeOptions<3>,
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        let shape = calculate_conv_transpose_output_shape(
            &x.shape,
            &weight.shape,
            &options.stride,
            &options.padding,
            &options.padding_out,
            &options.dilation,
            options.groups,
        )
        .unwrap();
        let dtype = output_dtype(
            [
                Some(&x.dtype),
                Some(&weight.dtype),
                bias.as_ref().map(|b| &b.dtype),
            ]
            .iter()
            .filter_map(|&d| d),
        )
        .unwrap();
        let out = TensorIr::uninit(new_id(), shape, dtype);

        ConvTranspose3dOpIr {
            x,
            weight,
            bias,
            options: options.into(),
            out,
        }
    }
}

impl AvgPool1dOpIr {
    pub fn create(
        x: TensorIr,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        let shape =
            calculate_pool_output_shape(&x.shape, &[kernel_size], &[stride], &[padding], &[1])
                .unwrap();
        let out = TensorIr::uninit(new_id(), shape, x.dtype);

        AvgPool1dOpIr {
            x,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            out,
        }
    }
}

impl AvgPool2dOpIr {
    pub fn create(
        x: TensorIr,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        let shape = calculate_pool_output_shape(&x.shape, &kernel_size, &stride, &padding, &[1, 1])
            .unwrap();
        let out = TensorIr::uninit(new_id(), shape, x.dtype);

        AvgPool2dOpIr {
            x,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            out,
        }
    }
}

impl AvgPool1dBackwardOpIr {
    pub fn create(
        x: TensorIr,
        grad: TensorIr,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        let out = TensorIr::uninit(new_id(), x.shape.clone(), x.dtype);

        AvgPool1dBackwardOpIr {
            x,
            grad,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            out,
        }
    }
}

impl AvgPool2dBackwardOpIr {
    pub fn create(
        x: TensorIr,
        grad: TensorIr,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        let out = TensorIr::uninit(new_id(), x.shape.clone(), x.dtype);

        AvgPool2dBackwardOpIr {
            x,
            grad,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            out,
        }
    }
}

impl AdaptiveAvgPool1dOpIr {
    pub fn create(x: TensorIr, output_size: usize, new_id: impl FnOnce() -> TensorId) -> Self {
        let shape = Shape::new([x.shape[0], x.shape[1], output_size]);
        let out = TensorIr::uninit(new_id(), shape, x.dtype);

        AdaptiveAvgPool1dOpIr {
            x,
            output_size,
            out,
        }
    }
}

impl AdaptiveAvgPool2dOpIr {
    pub fn create(x: TensorIr, output_size: [usize; 2], new_id: impl FnOnce() -> TensorId) -> Self {
        let shape = Shape::new([x.shape[0], x.shape[1], output_size[0], output_size[1]]);
        let out = TensorIr::uninit(new_id(), shape, x.dtype);

        AdaptiveAvgPool2dOpIr {
            x,
            output_size,
            out,
        }
    }
}

impl AdaptiveAvgPool1dBackwardOpIr {
    pub fn create(x: TensorIr, grad: TensorIr, new_id: impl FnOnce() -> TensorId) -> Self {
        let out = TensorIr::uninit(new_id(), x.shape.clone(), x.dtype);

        AdaptiveAvgPool1dBackwardOpIr { x, grad, out }
    }
}

impl AdaptiveAvgPool2dBackwardOpIr {
    pub fn create(x: TensorIr, grad: TensorIr, new_id: impl FnOnce() -> TensorId) -> Self {
        let out = TensorIr::uninit(new_id(), x.shape.clone(), x.dtype);

        AdaptiveAvgPool2dBackwardOpIr { x, grad, out }
    }
}

impl MaxPool1dOpIr {
    pub fn create(
        x: TensorIr,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        let shape = calculate_pool_output_shape(
            &x.shape,
            &[kernel_size],
            &[stride],
            &[padding],
            &[dilation],
        )
        .unwrap();
        let out = TensorIr::uninit(new_id(), shape, x.dtype);

        MaxPool1dOpIr {
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            out,
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

impl MaxPool1dWithIndicesBackwardOpIr {
    pub fn create(
        x: TensorIr,
        grad: TensorIr,
        indices: TensorIr,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        let out = TensorIr::uninit(new_id(), x.shape.clone(), x.dtype);

        MaxPool1dWithIndicesBackwardOpIr {
            x,
            grad,
            indices,
            kernel_size,
            stride,
            padding,
            dilation,
            out,
        }
    }
}

impl MaxPool2dOpIr {
    pub fn create(
        x: TensorIr,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        let shape =
            calculate_pool_output_shape(&x.shape, &kernel_size, &stride, &padding, &dilation)
                .unwrap();
        let out = TensorIr::uninit(new_id(), shape, x.dtype);

        MaxPool2dOpIr {
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            out,
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

impl MaxPool2dWithIndicesBackwardOpIr {
    pub fn create(
        x: TensorIr,
        grad: TensorIr,
        indices: TensorIr,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        let out = TensorIr::uninit(new_id(), x.shape.clone(), x.dtype);

        MaxPool2dWithIndicesBackwardOpIr {
            x,
            grad,
            indices,
            kernel_size,
            stride,
            padding,
            dilation,
            out,
        }
    }
}

impl InterpolateOpIr {
    pub fn create(
        x: TensorIr,
        output_size: [usize; 2],
        options: InterpolateOptions,
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        let shape = Shape::new([x.shape[0], x.shape[1], output_size[0], output_size[1]]);
        let out = TensorIr::uninit(new_id(), shape, x.dtype);

        InterpolateOpIr {
            x,
            output_size,
            options: options.into(),
            out,
        }
    }
}

impl InterpolateBackwardOpIr {
    pub fn create(
        x: TensorIr,
        grad: TensorIr,
        output_size: [usize; 2],
        options: InterpolateOptions,
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        let out = TensorIr::uninit(new_id(), x.shape.clone(), x.dtype);

        InterpolateBackwardOpIr {
            x,
            grad,
            output_size,
            options: options.into(),
            out,
        }
    }
}

impl UnfoldOpIr {
    pub fn create(
        input: TensorIr,
        dim: usize,
        size: usize,
        step: usize,
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        let shape = calculate_unfold_shape(input.shape.clone(), dim, size, step);
        let out = TensorIr::uninit(new_id(), shape, input.dtype);

        UnfoldOpIr {
            input,
            dim,
            size,
            step,
            out,
        }
    }
}

impl CrossOpIr {
    pub fn create(
        lhs: TensorIr,
        rhs: TensorIr,
        dim: usize,
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        let shape = lhs.shape.broadcast(&rhs.shape).unwrap();
        let dtype = output_dtype([&lhs.dtype, &rhs.dtype]).unwrap();
        let out = TensorIr::uninit(new_id(), shape, dtype);

        CrossOpIr { lhs, rhs, dim, out }
    }
}

impl QuantizeOpIr {
    pub fn create(
        tensor: TensorIr,
        qparams: QuantizationParametersIr,
        scheme: QuantScheme,
        new_id: impl FnOnce() -> TensorId,
    ) -> Self {
        let out = TensorIr::uninit(new_id(), tensor.shape.clone(), DType::QFloat(scheme));

        QuantizeOpIr {
            tensor,
            qparams,
            scheme,
            out,
        }
    }
}

impl DequantizeOpIr {
    pub fn create(input: TensorIr, dtype: DType, new_id: impl FnOnce() -> TensorId) -> Self {
        let out = TensorIr::uninit(new_id(), input.shape.clone(), dtype);

        DequantizeOpIr { input, out }
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
