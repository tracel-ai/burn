use burn_backend::{DType, QTensorPrimitive};
use cubecl::quant::scheme::{QuantStore, QuantValue};
use cubecl::server::AllocationKind;

use crate::{CubeRuntime, ops::empty_qtensor, tensor::CubeTensor};

/// Make a jit tensor contiguous.
pub fn into_contiguous<R: CubeRuntime>(tensor: CubeTensor<R>) -> CubeTensor<R> {
    if tensor.is_contiguous() {
        return tensor;
    }

    if tensor.qparams.is_some() {
        return into_contiguous_quantized(tensor, AllocationKind::Contiguous);
    }

    let output = cubecl::std::tensor::into_contiguous(
        &tensor.client,
        &tensor.as_handle_ref(),
        tensor.dtype.into(),
    )
    .expect("Kernel to never fail");

    CubeTensor::new(
        tensor.client,
        output.handle,
        output.shape.into(),
        tensor.device,
        output.strides,
        tensor.dtype,
    )
}

/// Make a jit tensor contiguous with an aligned last stride. Tensor is considered already contiguous
/// if runtime can read it as is. This is equivalent in practice.
#[tracing::instrument(skip(tensor))]
pub fn into_contiguous_aligned<R: CubeRuntime>(tensor: CubeTensor<R>) -> CubeTensor<R> {
    if R::can_read_tensor(&tensor.shape, &tensor.strides) {
        return tensor;
    }

    if tensor.qparams.is_some() {
        return into_contiguous_quantized(tensor, AllocationKind::Optimized);
    }

    let output = cubecl::std::tensor::into_contiguous_pitched(
        &tensor.client,
        &tensor.as_handle_ref(),
        tensor.dtype.into(),
    )
    .expect("Kernel to never fail");

    CubeTensor::new(
        tensor.client,
        output.handle,
        output.shape.into(),
        tensor.device,
        output.strides,
        tensor.dtype,
    )
}

#[tracing::instrument(skip(tensor))]
fn into_contiguous_quantized<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    kind: AllocationKind,
) -> CubeTensor<R> {
    let scheme = tensor.scheme();
    let output = empty_qtensor(tensor.shape.clone(), *tensor.scheme(), &tensor.device, kind);
    let (values, scales) = tensor.quantized_handles().unwrap();
    let (out_values, out_scales) = output.quantized_handles().unwrap();

    match scheme.store {
        QuantStore::U32 => {
            cubecl::std::tensor::into_contiguous_packed_ref(
                &values.client,
                &values.as_handle_ref(),
                &out_values.as_handle_ref(),
                &tensor.shape,
                scheme.num_quants() as u32,
                DType::U32.into(),
            )
            .expect("Kernel to never fail");
        }
        // e2m1 is special because it has a native packed representation, `e2m1x2`.
        // It's internally stored as `u8` with a packing factor of 2.
        QuantStore::Native if scheme.value == QuantValue::E2M1 => {
            cubecl::std::tensor::into_contiguous_packed_ref(
                &values.client,
                &values.as_handle_ref(),
                &out_values.as_handle_ref(),
                &tensor.shape,
                2,
                DType::U8.into(),
            )
            .expect("Kernel to never fail");
        }
        QuantStore::Native => {
            cubecl::std::tensor::into_contiguous_ref(
                &values.client,
                &values.as_handle_ref(),
                &out_values.as_handle_ref(),
                values.dtype.into(),
            )
            .expect("Kernel to never fail");
        }
    }

    cubecl::std::tensor::into_contiguous_ref(
        &scales.client,
        &scales.as_handle_ref(),
        &out_scales.as_handle_ref(),
        scales.dtype.into(),
    )
    .expect("Kernel to never fail");

    output
}
