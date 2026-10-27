//! A tensor as cubek's tile-DSL routines read it: its values bound as they are, or, quantized,
//! packed under its scales.

use super::CubeTensor;
use crate::kernel::into_contiguous;
use burn_backend::DType;
use burn_backend::cubecl::dtype_to_storage_type;
use cubecl::tile::{Axis, launch::Refusal};
use cubek::matmul::kernels::tensor::{PackedTensor, PlainTensor, StoredTensor, WordsLaid};
use cubek::microkernel::matmul::stored::Operand;

/// What [`CubeTensor::stored`] declines of a quantized tensor: a scheme with no block scales along
/// the axis its words run along.
pub const SCALES_NOT_ALONG_WORDS: Refusal = Refusal::Declined {
    why: "the packed routines read block scales along the axis the words run along",
};

/// What [`CubeTensor::stored`] declines of a quantized tensor: a batch of matrices, or words other
/// than row-major or in storage tiles.
pub const PACKED_NOT_ONE_MATRIX_IN_ROWS: Refusal = Refusal::Declined {
    why: "the packed routines read one quantized matrix in row-major rows",
};

impl CubeTensor {
    /// This tensor bound as its values, at the element they are stored in.
    pub fn plain(self) -> PlainTensor {
        let elem = dtype_to_storage_type(self.dtype);
        PlainTensor::new(self.binding(), elem)
    }

    /// This tensor, one matrix over `axes` in the order its shape states them, as it is stored:
    /// plain, or, quantized, packed under its scales. Its words run along the axis its scheme
    /// packs; its block scales and its per-tensor scale, where its scheme has one, each end on a
    /// whole word.
    ///
    /// # Errors
    ///
    /// For a quantized tensor, [`SCALES_NOT_ALONG_WORDS`] and [`PACKED_NOT_ONE_MATRIX_IN_ROWS`].
    pub fn stored(self, axes: [Axis; 2]) -> Result<StoredTensor, Refusal> {
        let DType::QFloat(scheme) = self.dtype else {
            return Ok(StoredTensor::Plain(self.plain()));
        };
        let shape = self.meta.shape().clone();
        let operand = Operand::axis_of(&scheme, axes)
            .and_then(|axis| Operand::of_scheme(&scheme, axis, axes))
            .ok_or(SCALES_NOT_ALONG_WORDS)?;
        let global = self
            .global()
            .map(|scale| into_contiguous(scale).bound_to_whole_words().plain());
        let (words, block_scales) = self
            .quantized_handles()
            .expect("a quantized tensor holds its values and its scales");
        if shape.rank() != 2 || words.meta.is_tiled() || !words.is_contiguous() {
            return Err(PACKED_NOT_ONE_MATRIX_IN_ROWS);
        }
        Ok(StoredTensor::Packed(PackedTensor::new(
            words.plain(),
            WordsLaid::InRows,
            into_contiguous(block_scales).bound_to_whole_words().plain(),
            global,
            scheme,
            operand,
            [(axes[0], shape[0]), (axes[1], shape[1])],
        )))
    }

    /// This contiguous tensor with its binding ending on the `u32` word that holds its last
    /// element: the routines read scales a word at a time. A quantized buffer starts each region
    /// on a word, so the word's tail is allocated.
    fn bound_to_whole_words(mut self) -> Self {
        let bytes = (self.meta.num_elements() * self.dtype.size()) as u64;
        let end = self.handle.offset_start.unwrap_or(0) + bytes.next_multiple_of(4);
        self.handle.offset_end = Some(self.handle.size() - end);
        self
    }
}
