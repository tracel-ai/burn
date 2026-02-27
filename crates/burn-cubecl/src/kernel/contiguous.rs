use burn_backend::{DType, QTensorPrimitive, TensorMetadata};
use cubecl::quant::scheme::{QuantStore, QuantValue};
use cubecl::server::MemoryLayoutStrategy;

use crate::{CubeRuntime, ops::empty_qtensor, tensor::CubeTensor};

/// Make a jit tensor contiguous.
pub fn into_contiguous<R: CubeRuntime>(tensor: CubeTensor<R>) -> CubeTensor<R> {
    if tensor.is_contiguous() {
        return tensor;
    }

    if tensor.qparams.is_some() {
        return into_contiguous_quantized(tensor, MemoryLayoutStrategy::Contiguous);
    }

    let (client, device, dtype) = (
        tensor.client.clone(),
        tensor.device.clone(),
        tensor.dtype.clone(),
    );

    let output = cubecl::std::tensor::into_contiguous(&client, tensor.binding(), dtype.into());

    CubeTensor::new(
        client.clone(),
        output.handle,
        *output.metadata,
        device,
        dtype,
    )
}

/// Make a jit tensor contiguous with an aligned last stride. Tensor is considered already contiguous
/// if runtime can read it as is. This is equivalent in practice.
#[cfg_attr(
    feature = "tracing",
    tracing::instrument(level = "trace", skip(tensor))
)]
pub fn into_contiguous_aligned<R: CubeRuntime>(tensor: CubeTensor<R>) -> CubeTensor<R> {
    if R::can_read_tensor(tensor.meta.shape(), tensor.meta.strides()) {
        return tensor;
    }

    if tensor.qparams.is_some() {
        return into_contiguous_quantized(tensor, MemoryLayoutStrategy::Optimized);
    }

    let (client, device, dtype) = (
        tensor.client.clone(),
        tensor.device.clone(),
        tensor.dtype.clone(),
    );

    let output =
        cubecl::std::tensor::into_contiguous_pitched(&client, tensor.binding(), dtype.into());

    CubeTensor::new(
        client.clone(),
        output.handle,
        *output.metadata,
        device,
        dtype,
    )
}

#[cfg_attr(
    feature = "tracing",
    tracing::instrument(level = "trace", skip(tensor))
)]
fn into_contiguous_quantized<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    strategy: MemoryLayoutStrategy,
) -> CubeTensor<R> {
    let scheme = tensor.scheme();
    let output = empty_qtensor(tensor.shape(), *tensor.scheme(), &tensor.device, strategy);
    let (values, scales) = tensor.quantized_handles().unwrap();
    let (out_values, out_scales) = output.quantized_handles().unwrap();

    let (client, dtype_scales, dtype_value) = (
        scales.client.clone(),
        scales.dtype.clone(),
        values.dtype.clone(),
    );

    match scheme.store {
        QuantStore::PackedU32(packed_dim) => {
            cubecl::std::tensor::into_contiguous_packed_ref(
                &client,
                values.binding(),
                out_values.binding(),
                packed_dim,
                tensor.meta.shape(),
                scheme.num_quants(),
                DType::U32.into(),
            );
        }
        // e2m1 is special because it has a native packed representation, `e2m1x2`.
        // It's internally stored as `u8` with a packing factor of 2.
        QuantStore::PackedNative(packed_dim) if scheme.value == QuantValue::E2M1 => {
            cubecl::std::tensor::into_contiguous_packed_ref(
                &client,
                values.binding(),
                out_values.binding(),
                packed_dim,
                tensor.meta.shape(),
                scheme.num_quants(),
                DType::U8.into(),
            );
        }
        _ => {
            cubecl::std::tensor::copy_into(
                &client,
                values.binding(),
                out_values.binding(),
                dtype_value.into(),
            );
        }
    }

    cubecl::std::tensor::copy_into(
        &client,
        scales.binding(),
        out_scales.binding(),
        dtype_scales.into(),
    );

    output
}
