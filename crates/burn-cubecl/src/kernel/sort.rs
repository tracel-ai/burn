//! Accelerated sorting operations using cubek radix sort.

use crate::{CubeRuntime, ops::numeric::empty_device_contiguous_dtype, tensor::CubeTensor};
use burn_backend::{DType, Shape};
use cubecl::prelude::*;
use cubek::sort::{SortError, SortOrder, sort_keys, sort_pairs};
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
    let output = empty_device_contiguous_dtype(
        tensor.client.clone(),
        tensor.device.clone(),
        Shape::new([num_items]),
        tensor.dtype,
    );

    let client = &tensor.client;
    let keys = tensor.as_handle_ref();
    let out_ref = output.as_handle_ref();

    let res = match tensor.dtype {
        DType::U8 => sort_keys::<R, u8>(client, keys, out_ref, num_items, order),
        DType::I8 => sort_keys::<R, i8>(client, keys, out_ref, num_items, order),
        DType::U16 => sort_keys::<R, u16>(client, keys, out_ref, num_items, order),
        DType::I16 => sort_keys::<R, i16>(client, keys, out_ref, num_items, order),
        DType::U32 => sort_keys::<R, u32>(client, keys, out_ref, num_items, order),
        DType::I32 => sort_keys::<R, i32>(client, keys, out_ref, num_items, order),
        DType::U64 => sort_keys::<R, u64>(client, keys, out_ref, num_items, order),
        DType::I64 => sort_keys::<R, i64>(client, keys, out_ref, num_items, order),
        DType::F16 => sort_keys::<R, f16>(client, keys, out_ref, num_items, order),
        DType::BF16 => sort_keys::<R, bf16>(client, keys, out_ref, num_items, order),
        DType::Flex32 | DType::F32 => sort_keys::<R, f32>(client, keys, out_ref, num_items, order),
        DType::F64 => sort_keys::<R, f64>(client, keys, out_ref, num_items, order),
        // Bool/QFloat tensors don't have sort operations at the API level
        DType::Bool | DType::QFloat(_) => {
            unreachable!("Bool and QFloat tensors should not get here.")
        }
    };

    match res {
        Ok(_) => Ok(output),
        Err(err) => Err(err),
    }
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
    let values_out = empty_device_contiguous_dtype(
        tensor.client.clone(),
        tensor.device.clone(),
        Shape::new([num_items]),
        tensor.dtype,
    );
    let indices_out = empty_device_contiguous_dtype(
        tensor.client.clone(),
        tensor.device.clone(),
        Shape::new([num_items]),
        DType::U32,
    );

    // TODO: Create this with arrange, or even directly in sorting kernels.
    // Create indices [0, 1, 2, ...]
    let indices_data: Vec<u32> = (0..num_items as u32).collect();
    let indices_handle = tensor
        .client
        .create_from_slice(u32::as_bytes(&indices_data));
    let indices_in = CubeTensor::new_contiguous(
        tensor.client.clone(),
        tensor.device.clone(),
        Shape::new([num_items]),
        indices_handle,
        DType::U32,
    );

    let client = &tensor.client;
    let keys = tensor.as_handle_ref();
    let val_ref = values_out.as_handle_ref();
    let in_ref = indices_in.as_handle_ref();
    let out_ref = indices_out.as_handle_ref();

    let res = match tensor.dtype {
        DType::U8 => {
            sort_pairs::<R, u8, u32>(client, keys, val_ref, in_ref, out_ref, num_items, order)
        }
        DType::I8 => {
            sort_pairs::<R, i8, u32>(client, keys, val_ref, in_ref, out_ref, num_items, order)
        }
        DType::U16 => {
            sort_pairs::<R, u16, u32>(client, keys, val_ref, in_ref, out_ref, num_items, order)
        }
        DType::I16 => {
            sort_pairs::<R, i16, u32>(client, keys, val_ref, in_ref, out_ref, num_items, order)
        }
        DType::U32 => {
            sort_pairs::<R, u32, u32>(client, keys, val_ref, in_ref, out_ref, num_items, order)
        }
        DType::I32 => {
            sort_pairs::<R, i32, u32>(client, keys, val_ref, in_ref, out_ref, num_items, order)
        }
        DType::U64 => {
            sort_pairs::<R, u64, u32>(client, keys, val_ref, in_ref, out_ref, num_items, order)
        }
        DType::I64 => {
            sort_pairs::<R, i64, u32>(client, keys, val_ref, in_ref, out_ref, num_items, order)
        }
        DType::F16 => {
            sort_pairs::<R, f16, u32>(client, keys, val_ref, in_ref, out_ref, num_items, order)
        }
        DType::BF16 => {
            sort_pairs::<R, bf16, u32>(client, keys, val_ref, in_ref, out_ref, num_items, order)
        }
        DType::Flex32 | DType::F32 => {
            sort_pairs::<R, f32, u32>(client, keys, val_ref, in_ref, out_ref, num_items, order)
        }
        DType::F64 => {
            sort_pairs::<R, f64, u32>(client, keys, val_ref, in_ref, out_ref, num_items, order)
        }
        // Bool/QFloat tensors don't have sort operations at the API level
        DType::Bool | DType::QFloat(_) => {
            unreachable!("Bool and QFloat tensors should not get here.")
        }
    };

    match res {
        Ok(_) => Ok((values_out, indices_out)),
        Err(err) => Err(err),
    }
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
