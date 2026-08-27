use alloc::vec::Vec;

use burn_std::{DType, PadMode, Shape, Slice};

use crate::{
    Backend, Scalar, TensorMetadata,
    tensor::{FloatTensor, IntTensor},
};

fn slices(shape: &Shape, dim: usize, start: usize, len: usize) -> Vec<Slice> {
    shape
        .iter()
        .enumerate()
        .map(|(axis, size)| {
            if axis == dim {
                Slice::from(start..start + len)
            } else {
                Slice::from(0..*size)
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn pad<B, T, Full, Zeros, SliceFn, SliceAssign, Flip, Repeat>(
    tensor: T,
    padding: &[(usize, usize)],
    mode: PadMode,
    full: Full,
    zeros: Zeros,
    slice: SliceFn,
    slice_assign: SliceAssign,
    flip: Flip,
    repeat: Repeat,
) -> T
where
    B: Backend,
    T: Clone + TensorMetadata<Device = B::Device>,
    Full: Fn(Shape, Scalar, &B::Device, DType) -> T,
    Zeros: Fn(Shape, &B::Device, DType) -> T,
    SliceFn: Fn(T, &[Slice]) -> T,
    SliceAssign: Fn(T, &[Slice], T) -> T,
    Flip: Fn(T, &[usize]) -> T,
    Repeat: Fn(T, usize, usize) -> T,
{
    assert_eq!(
        padding.len(),
        tensor.shape().num_dims(),
        "padding must have one pair per dimension"
    );

    let original_shape = tensor.shape();
    match mode {
        PadMode::Constant(value) => {
            let mut output_dims = original_shape.to_vec();
            for (dim, (before, after)) in padding.iter().enumerate() {
                output_dims[dim] += before + after;
            }
            let output_shape = Shape::from(output_dims);
            let device = tensor.device();
            let dtype = tensor.dtype();
            let center: Vec<_> = original_shape
                .iter()
                .enumerate()
                .map(|(dim, size)| Slice::from(padding[dim].0..padding[dim].0 + size))
                .collect();
            let output = full(output_shape, Scalar::Float(value as f64), &device, dtype);
            slice_assign(output, &center, tensor)
        }
        PadMode::Reflect | PadMode::Edge => {
            for (dim, (before, after)) in padding.iter().copied().enumerate() {
                if matches!(mode, PadMode::Reflect) {
                    assert!(
                        before < original_shape[dim] && after < original_shape[dim],
                        "Reflect padding must be less than dimension size"
                    );
                } else if before != 0 || after != 0 {
                    assert!(
                        original_shape[dim] != 0,
                        "cannot apply edge padding to an empty dimension"
                    );
                }
            }

            let mut result = tensor;
            for (dim, (before, after)) in padding.iter().copied().enumerate() {
                if before == 0 && after == 0 {
                    continue;
                }
                let shape = result.shape();
                let size = shape[dim];
                let mut output_dims = shape.to_vec();
                output_dims[dim] += before + after;
                let output_shape = Shape::from(output_dims);
                let device = result.device();
                let dtype = result.dtype();
                let output = zeros(output_shape.clone(), &device, dtype);
                let mut output = slice_assign(
                    output,
                    &slices(&output_shape, dim, before, size),
                    result.clone(),
                );
                if before > 0 {
                    let (start, len) = if matches!(mode, PadMode::Reflect) {
                        (1, before)
                    } else {
                        (0, 1)
                    };
                    let value = slice(result.clone(), &slices(&shape, dim, start, len));
                    let value = if matches!(mode, PadMode::Reflect) {
                        flip(value, &[dim])
                    } else {
                        repeat(value, dim, before)
                    };
                    output = slice_assign(output, &slices(&output_shape, dim, 0, before), value);
                }
                if after > 0 {
                    let (start, len) = if matches!(mode, PadMode::Reflect) {
                        (size - after - 1, after)
                    } else {
                        (size - 1, 1)
                    };
                    let value = slice(result, &slices(&shape, dim, start, len));
                    let value = if matches!(mode, PadMode::Reflect) {
                        flip(value, &[dim])
                    } else {
                        repeat(value, dim, after)
                    };
                    output = slice_assign(
                        output,
                        &slices(&output_shape, dim, before + size, after),
                        value,
                    );
                }
                result = output;
            }
            result
        }
    }
}

pub(crate) fn float_pad<B: Backend>(
    tensor: FloatTensor<B>,
    padding: &[(usize, usize)],
    mode: PadMode,
) -> FloatTensor<B> {
    pad::<B, _, _, _, _, _, _, _>(
        tensor,
        padding,
        mode,
        |shape, value, device, dtype| B::float_full(shape, value, device, dtype.into()),
        |shape, device, dtype| B::float_zeros(shape, device, dtype.into()),
        B::float_slice,
        B::float_slice_assign,
        B::float_flip,
        B::float_repeat_dim,
    )
}

pub(crate) fn int_pad<B: Backend>(
    tensor: IntTensor<B>,
    padding: &[(usize, usize)],
    mode: PadMode,
) -> IntTensor<B> {
    pad::<B, _, _, _, _, _, _, _>(
        tensor,
        padding,
        mode,
        |shape, value, device, dtype| B::int_full(shape, value, device, dtype.into()),
        |shape, device, dtype| B::int_zeros(shape, device, dtype.into()),
        B::int_slice,
        B::int_slice_assign,
        B::int_flip,
        B::int_repeat_dim,
    )
}
