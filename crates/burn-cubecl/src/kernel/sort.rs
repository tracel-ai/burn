//! Accelerated sorting operations using cubek radix sort.

use crate::{CubeRuntime, ops::numeric::empty_device_contiguous_dtype, tensor::CubeTensor};
use burn_backend::{DType, Shape};
use cubecl::prelude::*;
use cubek::sort::{Radix, SortError, SortKey, SortOrder, sort_keys, sort_pairs};
use half::{bf16, f16};

/// Sort a 1D tensor using radix sort.
///
/// Panics if tensor is not 1D.
pub fn sort_1d<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    descending: bool,
) -> Result<CubeTensor<R>, SortError> {
    assert!(tensor.shape.num_dims() == 1, "sort_1d requires 1D tensor");

    let order = if descending {
        SortOrder::Descending
    } else {
        SortOrder::Ascending
    };

    match tensor.dtype {
        DType::U8 => sort_1d_impl::<R, u8>(tensor, order),
        DType::I8 => sort_1d_impl::<R, i8>(tensor, order),
        DType::U16 => sort_1d_impl::<R, u16>(tensor, order),
        DType::I16 => sort_1d_impl::<R, i16>(tensor, order),
        DType::U32 => sort_1d_impl::<R, u32>(tensor, order),
        DType::I32 => sort_1d_impl::<R, i32>(tensor, order),
        DType::U64 => sort_1d_impl::<R, u64>(tensor, order),
        DType::I64 => sort_1d_impl::<R, i64>(tensor, order),
        DType::F16 => sort_1d_impl::<R, f16>(tensor, order),
        DType::BF16 => sort_1d_impl::<R, bf16>(tensor, order),
        DType::Flex32 | DType::F32 => sort_1d_impl::<R, f32>(tensor, order),
        DType::F64 => sort_1d_impl::<R, f64>(tensor, order),
        // Bool/QFloat tensors don't have sort operations at the API level
        DType::Bool | DType::QFloat(_) => {
            unreachable!("Bool and QFloat tensors should not reach sort_1d")
        }
    }
}

/// Sort a 1D tensor with indices using radix sort.
///
/// Panics if tensor is not 1D.
#[allow(clippy::type_complexity)]
pub fn sort_with_indices_1d<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    descending: bool,
) -> Result<(CubeTensor<R>, CubeTensor<R>), SortError> {
    assert!(
        tensor.shape.num_dims() == 1,
        "sort_with_indices_1d requires 1D tensor"
    );

    let order = if descending {
        SortOrder::Descending
    } else {
        SortOrder::Ascending
    };

    match tensor.dtype {
        DType::U8 => sort_with_indices_1d_impl::<R, u8>(tensor, order),
        DType::I8 => sort_with_indices_1d_impl::<R, i8>(tensor, order),
        DType::U16 => sort_with_indices_1d_impl::<R, u16>(tensor, order),
        DType::I16 => sort_with_indices_1d_impl::<R, i16>(tensor, order),
        DType::U32 => sort_with_indices_1d_impl::<R, u32>(tensor, order),
        DType::I32 => sort_with_indices_1d_impl::<R, i32>(tensor, order),
        DType::U64 => sort_with_indices_1d_impl::<R, u64>(tensor, order),
        DType::I64 => sort_with_indices_1d_impl::<R, i64>(tensor, order),
        DType::F16 => sort_with_indices_1d_impl::<R, f16>(tensor, order),
        DType::BF16 => sort_with_indices_1d_impl::<R, bf16>(tensor, order),
        DType::Flex32 | DType::F32 => sort_with_indices_1d_impl::<R, f32>(tensor, order),
        DType::F64 => sort_with_indices_1d_impl::<R, f64>(tensor, order),
        // Bool/QFloat tensors don't have sort operations at the API level
        DType::Bool | DType::QFloat(_) => {
            unreachable!("Bool and QFloat tensors should not reach sort_with_indices_1d")
        }
    }
}

/// Argsort a 1D tensor using radix sort.
///
/// Panics if tensor is not 1D.
pub fn argsort_1d<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    descending: bool,
) -> Result<CubeTensor<R>, SortError> {
    sort_with_indices_1d(tensor, descending).map(|(_, indices)| indices)
}

fn sort_1d_impl<R: CubeRuntime, K: SortKey>(
    tensor: CubeTensor<R>,
    order: SortOrder,
) -> Result<CubeTensor<R>, SortError>
where
    K::Radix: Radix + SortKey<Radix = K::Radix>,
{
    let num_items = tensor.shape[0];

    let output = empty_device_contiguous_dtype(
        tensor.client.clone(),
        tensor.device.clone(),
        Shape::new([num_items]),
        tensor.dtype,
    );

    sort_keys::<R, K>(
        &tensor.client,
        tensor.as_handle_ref(),
        output.as_handle_ref(),
        num_items,
        order,
    )?;

    Ok(output)
}

fn sort_with_indices_1d_impl<R: CubeRuntime, K: SortKey + CubePrimitive>(
    tensor: CubeTensor<R>,
    order: SortOrder,
) -> Result<(CubeTensor<R>, CubeTensor<R>), SortError>
where
    K::Radix: Radix + SortKey<Radix = K::Radix>,
{
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

    sort_pairs::<R, K, u32>(
        &tensor.client,
        tensor.as_handle_ref(),
        values_out.as_handle_ref(),
        indices_in.as_handle_ref(),
        indices_out.as_handle_ref(),
        num_items,
        order,
    )?;

    Ok((values_out, indices_out))
}
