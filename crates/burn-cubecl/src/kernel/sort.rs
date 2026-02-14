//! Accelerated sorting operations using cubek radix sort.

use crate::{CubeRuntime, tensor::CubeTensor};
use burn_backend::{DType, Shape};
use cubek::sort::{SortError, SortOrder, SortValues, sort as cubek_sort};
use half::{bf16, f16};

/// Sort a tensor using radix sort.
///
/// Panics if tensor is not 1D.
pub fn sort<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    order: SortOrder,
) -> Result<CubeTensor<R>, SortError> {
    assert!(tensor.shape.num_dims() == 1, "sort requires 1D tensor");

    let num_items = tensor.shape[0];
    let client = &tensor.client;
    let keys = tensor.as_handle_ref();

    let output = match tensor.dtype {
        DType::U8 => cubek_sort::<R, u8>(client, keys, SortValues::None, order),
        DType::I8 => cubek_sort::<R, i8>(client, keys, SortValues::None, order),
        DType::U16 => cubek_sort::<R, u16>(client, keys, SortValues::None, order),
        DType::I16 => cubek_sort::<R, i16>(client, keys, SortValues::None, order),
        DType::U32 => cubek_sort::<R, u32>(client, keys, SortValues::None, order),
        DType::I32 => cubek_sort::<R, i32>(client, keys, SortValues::None, order),
        DType::U64 => cubek_sort::<R, u64>(client, keys, SortValues::None, order),
        DType::I64 => cubek_sort::<R, i64>(client, keys, SortValues::None, order),
        DType::F16 => cubek_sort::<R, f16>(client, keys, SortValues::None, order),
        DType::BF16 => cubek_sort::<R, bf16>(client, keys, SortValues::None, order),
        DType::Flex32 | DType::F32 => cubek_sort::<R, f32>(client, keys, SortValues::None, order),
        DType::F64 => cubek_sort::<R, f64>(client, keys, SortValues::None, order),
        DType::Bool | DType::QFloat(_) => {
            unreachable!("Bool and QFloat tensors should not get here.")
        }
    }?;

    Ok(CubeTensor::new_contiguous(
        tensor.client.clone(),
        tensor.device.clone(),
        Shape::new([num_items]),
        output.keys,
        tensor.dtype,
    ))
}

/// Sort a 1D tensor with indices using radix sort.
///
/// Panics if tensor is not 1D.
#[allow(clippy::type_complexity)]
pub fn sort_with_indices<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    order: SortOrder,
) -> Result<(CubeTensor<R>, CubeTensor<R>), SortError> {
    assert!(
        tensor.shape.num_dims() == 1,
        "sort_with_indices_1d requires 1D tensor"
    );

    let num_items = tensor.shape[0];
    let client = &tensor.client;
    let keys = tensor.as_handle_ref();

    // Use SortValues::Indices - generates [0, 1, 2, ...] implicitly in the kernel
    let output = match tensor.dtype {
        DType::U8 => cubek_sort::<R, u8>(client, keys, SortValues::Indices, order),
        DType::I8 => cubek_sort::<R, i8>(client, keys, SortValues::Indices, order),
        DType::U16 => cubek_sort::<R, u16>(client, keys, SortValues::Indices, order),
        DType::I16 => cubek_sort::<R, i16>(client, keys, SortValues::Indices, order),
        DType::U32 => cubek_sort::<R, u32>(client, keys, SortValues::Indices, order),
        DType::I32 => cubek_sort::<R, i32>(client, keys, SortValues::Indices, order),
        DType::U64 => cubek_sort::<R, u64>(client, keys, SortValues::Indices, order),
        DType::I64 => cubek_sort::<R, i64>(client, keys, SortValues::Indices, order),
        DType::F16 => cubek_sort::<R, f16>(client, keys, SortValues::Indices, order),
        DType::BF16 => cubek_sort::<R, bf16>(client, keys, SortValues::Indices, order),
        DType::Flex32 | DType::F32 => {
            cubek_sort::<R, f32>(client, keys, SortValues::Indices, order)
        }
        DType::F64 => cubek_sort::<R, f64>(client, keys, SortValues::Indices, order),
        DType::Bool | DType::QFloat(_) => {
            unreachable!("Bool and QFloat tensors should not get here.")
        }
    }?;

    let values_out = CubeTensor::new_contiguous(
        tensor.client.clone(),
        tensor.device.clone(),
        Shape::new([num_items]),
        output.keys,
        tensor.dtype,
    );

    let indices_out = CubeTensor::new_contiguous(
        tensor.client.clone(),
        tensor.device.clone(),
        Shape::new([num_items]),
        output
            .values
            .expect("SortValues::Indices should produce values"),
        DType::U32,
    );

    Ok((values_out, indices_out))
}

/// Argsort a tensor using radix sort.
///
/// Panics if tensor is not 1D.
pub fn argsort<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    order: SortOrder,
) -> Result<CubeTensor<R>, SortError> {
    sort_with_indices(tensor, order).map(|(_, indices)| indices)
}
