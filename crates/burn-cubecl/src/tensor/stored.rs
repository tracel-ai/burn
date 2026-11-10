//! A tensor as cubek's tile-DSL routines read it: its values bound as they are, or, quantized,
//! packed under its scales.

use super::CubeTensor;
use burn_backend::DType;
use burn_backend::cubecl::dtype_to_storage_type;
use cubecl::tile::{Axis, launch::Refusal};
use cubek::matmul::kernels::tensor::{PackedTensor, PlainTensor, StoredTensor};

impl CubeTensor {
    /// This tensor bound as its values, at the element they are stored in.
    pub fn plain(self) -> PlainTensor {
        let elem = dtype_to_storage_type(self.dtype);
        PlainTensor::new(self.stored_binding(), elem)
    }

    /// This tensor, one matrix over `axes` in the order its shape states them, as it is stored:
    /// plain, or, quantized, packed under its scales ([`PackedTensor::of_quantized`]).
    ///
    /// # Errors
    ///
    /// For a quantized tensor, what [`PackedTensor::of_quantized`] declines.
    pub fn stored(self, axes: [Axis; 2]) -> Result<StoredTensor, Refusal> {
        let DType::QFloat(scheme) = self.dtype else {
            return Ok(StoredTensor::Plain(self.plain()));
        };
        let block_scales = self
            .scales()
            .expect("a quantized tensor holds its scales")
            .plain();
        let global = self.global().map(CubeTensor::plain);
        let client = self.client.clone();
        PackedTensor::of_quantized(
            &client,
            self.stored_binding(),
            scheme,
            block_scales,
            global,
            axes,
        )
        .map(StoredTensor::Packed)
    }
}

// Quantizes on a device: the Linux runners have no GPU for the test runtime's wgpu.
#[cfg(all(test, any(feature = "cuda", feature = "vulkan", feature = "metal")))]
mod tests {
    use burn_backend::{
        TensorData,
        ops::QTensorOps,
        quantization::{QuantScheme, QuantValue, ScaleDtype},
    };
    use cubecl::quant::scheme::QuantStore;
    use cubek::matmul::kernels::tensor::StoredTensor;
    use cubek::std::matmul_axes::{K, N};

    use crate::{
        CubeBackend, CubeDevice,
        ops::{from_data, swap_dims},
        tensor::CubeTensor,
    };

    /// A tied embedding read as the vocabulary head: `[n, k]` packed along `k`, transposed to
    /// `[k, n]`, its words now running along the view's first axis. The matmul reads it as stored,
    /// as it reads the projections; refused, it would dequantize the whole head on every call.
    #[test]
    fn a_transposed_packed_matrix_is_read_as_stored() {
        let device = CubeDevice::default();
        let (n, k) = (96, 64);
        let values: Vec<f32> = (0..n * k).map(|i| (i % 29) as f32 / 29. - 0.5).collect();
        let embedding: CubeTensor = from_data(TensorData::new(values, [n, k]), &device);
        let scheme = QuantScheme::default()
            .with_value(QuantValue::E2M1)
            .per_block([16], ScaleDtype::UE4M3)
            .per_tensor(ScaleDtype::F32)
            .with_store(QuantStore::PackedU32(0));
        let head = swap_dims(CubeBackend::quantize_dynamic(embedding, &scheme), 0, 1);

        let stored = head.stored([K, N]);
        assert!(
            matches!(stored, Ok(StoredTensor::Packed(_))),
            "a transposed packed matrix is read packed, got {stored:?}"
        );
    }
}
