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
        PlainTensor::new(self.binding(), elem)
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
        PackedTensor::of_quantized(&client, self.binding(), scheme, block_scales, global, axes)
            .map(StoredTensor::Packed)
    }
}
