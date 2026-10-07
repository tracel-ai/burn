//! Quantization data representation.

// Re-exported types
pub use cubecl_common::quant::scheme::{
    BlockScale, BlockSize, QuantMode, QuantScheme, QuantStore, QuantValue, ScaleDtype,
};

/// Alignment (in bytes) for quantization parameters in serialized tensor data.
///
/// NOTE: This is currently f32-based since scales were originally always f32.
/// With `ScaleDtype` now supporting different precisions (F16, BF16, etc.),
/// this alignment may need to be revisited in the future.
pub const QPARAM_ALIGN: usize = core::mem::align_of::<f32>();

use alloc::{vec, vec::Vec};
use core::any::TypeId;
use cubecl_common::e4m3;
use cubecl_common::quant::scheme::{f32_to_ue8m0, ue8m0_to_f32};
use num_traits::PrimInt;
use serde::{Deserialize, Serialize};

use crate::{DType, Metadata, Shape, bytes::Bytes};

/// Configuration for a device quantization behavior.
///
/// This configuration determines how tensors are quantized and how quantization rules
/// propagate through operations on a given device. It is applied once during device
/// initialization. See also the [device settings](crate::DeviceSettings).
#[derive(new, Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct QuantConfig {
    /// Defines how a tensor is quantized.
    pub scheme: QuantScheme,
    /// How quantization is propagated during computation.
    pub propagation: QuantPropagation,
    // NOTE: accumulation is currently unused, only scheme and propagation have an impact
    // /// The precision used for the accumulation in various kernels.
    // pub acc: QuantAcc,
}

#[derive(
    Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default,
)]
/// The precision of accumulating elements.
pub enum QuantAcc {
    /// Full precision.
    #[default]
    F32,
    /// Half precision.
    F16,
    /// bfloat16 precision.
    BF16,
}

/// Calibration method used to compute the quantization range mapping.
pub enum Calibration {
    /// Computes quantization range mapping based on the min and max values.
    MinMax,
    /// Absolute-mean calibration for BitNet b1.58-style `{-1, 0, +1}` weight quantization.
    ///
    /// The range is `[-γ, +γ]` where γ = `mean(|W|)` per tensor or per block (BitNet b1.58
    /// §3.1). Use with `QuantValue::Q2S` and `QuantStore::PackedU32` for 2-bit packed storage.
    AbsMean,
}

/// Specify if the output of an operation is quantized using the scheme of the input
/// or returned unquantized.
#[derive(
    Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default,
)]
pub enum QuantPropagation {
    /// The output is quantized using the scheme of the input.
    Propagate,
    /// The output is not quantized.
    #[default]
    Inhibit,
}

/// The quantization tensor data parameters.
#[derive(Clone, Debug)]
pub struct QParams<S> {
    /// The scaling factor.
    pub scales: S,
    /// The per-tensor scale [`scales`](Self::scales) are relative to, for a two-level scheme.
    pub global: Option<S>,
}

/// Scales recovered from a quantized byte buffer.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodedScales {
    /// One scale per block, or a single entry for a per-tensor level.
    pub block: Vec<f32>,
    /// The per-tensor scale, for a level that carries one.
    pub global: Option<f32>,
}

/// A quantization parameter tensor descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QParamTensor {
    /// Start of the tensor in the buffer
    pub offset_start: usize,
    /// Offset of tensor end from the end of the buffer
    pub offset_end: usize,
    /// Metadata of the tensor
    pub metadata: Metadata,
    /// Data type of the tensor
    pub dtype: DType,
}

/// Whether a backend can quantize against this scheme's scales.
///
/// A backend answers `supports_dtype` with this, so an unsupported scheme is declined where it is
/// chosen rather than at the first quantize. Each condition is asserted again where it is relied
/// on, so bypassing this reports the specific rule rather than this general one.
pub fn quantizable(scheme: &QuantScheme) -> bool {
    match (scheme.block_scale(), global_scale_dtype(scheme)) {
        (Some(block), Some(global)) => {
            // The per-tensor scale is the largest block scale over the block dtype's maximum, so a
            // block dtype reaching f32's range drives it subnormal. Any precision the per-tensor
            // scale itself loses becomes a mismatch applied to every block.
            block.dtype.max_representable() <= crate::f16::MAX.to_f32() && global == ScaleDtype::F32
        }
        _ => true,
    }
}

/// The dtype of the per-tensor scale block scales are normalized against, for a two-level scheme.
///
/// [`QuantScheme::tensor_scale`] answers for a per-tensor scheme too, where the scale is the whole
/// reconstruction rather than a factor over block scales.
pub fn global_scale_dtype(scheme: &QuantScheme) -> Option<ScaleDtype> {
    scheme.block_scale().and(scheme.tensor_scale())
}

/// Calculate the shape of the block scale grid for a given tensor and scheme.
///
/// A two-level scheme's per-tensor scale is not part of this grid.
pub fn params_shape(data_shape: &Shape, scheme: &QuantScheme) -> Shape {
    match scheme.block_size() {
        None => Shape::new([1]),
        Some(block_size) => Shape::from(block_size.num_blocks(data_shape.as_slice())),
    }
}

/// Which block each element of a tensor falls in, for a block scheme. A block is a rectangle, so
/// its members are a run of the flat storage only when it spans the trailing dimension; anything
/// walking values against block scales asks here rather than chunking.
#[derive(Debug, Clone)]
pub struct BlockLayout {
    shape: Shape,
    block: Vec<u8>,
    blocks: Shape,
}

impl BlockLayout {
    /// How `block` tiles a tensor of `shape`.
    pub fn new(shape: &Shape, block: &BlockSize) -> Self {
        Self {
            shape: shape.clone(),
            block: block.to_dim_vec(shape.num_dims()),
            blocks: Shape::from(block.num_blocks(shape.as_slice())),
        }
    }

    /// How many blocks the tensor holds, which is how many block scales it has.
    pub fn num_blocks(&self) -> usize {
        self.blocks.num_elements()
    }

    /// Whether every dimension is a whole number of blocks.
    pub fn divides(&self) -> bool {
        self.shape
            .iter()
            .zip(&self.block)
            .all(|(&dim, &extent)| dim.is_multiple_of(extent as usize))
    }

    /// The row-major index of the block holding row-major element `index`.
    pub fn block_of(&self, mut index: usize) -> usize {
        let mut block = 0;
        let mut stride = 1;
        for dim in (0..self.shape.num_dims()).rev() {
            let coordinate = index % self.shape[dim];
            index /= self.shape[dim];
            block += coordinate / self.block[dim] as usize * stride;
            stride *= self.blocks[dim];
        }
        block
    }
}

/// Quantized data bytes representation.
///
/// # Notes
/// 1) The quantized values are packed into 32-bit unsigned integers. For example, int8
///    quantized values pack 4 grouped values into a single `u32`. When unpacking these values,
///    we make sure to retrieve only the meaningful values (and ignore the alignment padding).
/// 2) Quantization parameters are appended to the tensor data.
///    As such, the last bytes always correspond to the scale parameter.
///    If the quantization scheme includes an offset (zero-point) parameter, it is next to last.
pub struct QuantizedBytes {
    /// The quantized values and quantization parameters represented as bytes.
    pub bytes: Bytes,
    /// The quantization scheme.
    pub scheme: QuantScheme,
    /// The shape of the quantized tensor. The block count, and so the scale count, follows from
    /// it per axis: a block that does not span the trailing dimension is not a run of elements.
    pub shape: Shape,
}

impl QuantizedBytes {
    /// Creates a new quantized bytes representation.
    ///
    /// `global` is the per-tensor scale, required by a two-level scheme and rejected by a
    /// one-level one.
    pub fn new<E: bytemuck::CheckedBitPattern + bytemuck::NoUninit>(
        value: Vec<E>,
        shape: impl Into<Shape>,
        scheme: QuantScheme,
        scales: &[f32],
        global: Option<f32>,
    ) -> Self {
        let shape = shape.into();
        assert_eq!(
            value.len(),
            shape.num_elements(),
            "{} quantized values do not fill a tensor of shape {shape:?}",
            value.len()
        );
        // Only used for 8-bit quantization data comparison in tests
        if TypeId::of::<E>() != TypeId::of::<i8>() {
            panic!("Invalid quantized type");
        }

        // Re-interpret `Vec<E>` as `Vec<i8>` with `Vec::from_raw_parts`
        let i8s: Vec<i8> = bytemuck::allocation::cast_vec(value);
        let mut bytes = match scheme.store {
            QuantStore::PackedU32(packed_dim) => {
                Bytes::from_elems(pack_u32(&i8s, &shape, packed_dim, scheme.value.size_bits()))
            }
            QuantStore::Native | QuantStore::PackedNative(_) => Bytes::from_elems(i8s),
        };

        let scales = match scheme.block_size() {
            None => &scales[..1],
            Some(_) => scales,
        };
        let scale_bytes = encode_scales(scales, scheme.scale_dtype());
        bytes.extend_from_byte_slice_aligned(scale_bytes.as_slice(), QPARAM_ALIGN);

        // Last, so a reader can peel it off the end before the block scales it normalizes.
        match (global_scale_dtype(&scheme), global) {
            (Some(dtype), Some(global)) => {
                // Encoding the per-tensor scale narrower would round it, and the block scales were
                // normalized against the unrounded one.
                assert_eq!(
                    dtype,
                    ScaleDtype::F32,
                    "a two-level scheme stores its per-tensor scale as f32, got {scheme:?}"
                );
                let global_bytes = encode_scales(&[global], dtype);
                bytes.extend_from_byte_slice_aligned(global_bytes.as_slice(), QPARAM_ALIGN);
            }
            (Some(_), None) => panic!("{scheme:?} requires a per-tensor scale"),
            (None, Some(_)) => panic!("{scheme:?} does not take a per-tensor scale"),
            (None, None) => {}
        }

        Self {
            bytes,
            scheme,
            shape,
        }
    }

    /// The number of quantized elements.
    pub fn num_elements(&self) -> usize {
        self.shape.num_elements()
    }

    /// Returns the int8 quantized values with the quantization parameters.
    pub fn into_vec_i8(self) -> (Vec<i8>, DecodedScales) {
        let scheme = self.scheme;
        let (values, (qparams, num_params)) = self.split_values_off();

        // Laid out as `[block scale, ...]` optionally followed by the per-tensor scale.
        let global_bytes = global_scale_size(&scheme);
        let block_end = qparams
            .len()
            .checked_sub(global_bytes)
            .expect("quantized parameter buffer is shorter than the scheme's global scale");
        let block_start = block_end
            .checked_sub(scale_size(scheme.scale_dtype()) * num_params)
            .expect("quantized parameter buffer is shorter than the scheme's block scales");

        let block = decode_scales(&qparams[block_start..block_end], scheme.scale_dtype());
        let global =
            global_scale_dtype(&scheme).map(|dtype| decode_scales(&qparams[block_end..], dtype)[0]);

        (values, DecodedScales { block, global })
    }

    /// Splits the quantized values of the tensor from the quantization parameters.
    ///
    /// Returns the values in i8 and a newly allocated vector containing the
    /// quantization parameter bytes.
    fn split_values_off(self) -> (Vec<i8>, (Vec<u8>, usize)) {
        let num_params = params_shape(&self.shape, &self.scheme).num_elements();
        let scale_bytes =
            scale_size(self.scheme.scale_dtype()) * num_params + global_scale_size(&self.scheme);
        let num_elements = self.num_elements();
        let (shape, scheme) = (self.shape, self.scheme);

        let mut stored = read_bytes_to_i8(self.bytes);
        let split_at = stored
            .len()
            .checked_sub(scale_bytes)
            .expect("quantized tensor data is shorter than its scheme's parameters");
        let qparams = bytemuck::cast_vec(stored.split_off(split_at));

        let values = match scheme.store {
            // Values and scales may be separated by alignment padding.
            QuantStore::Native => {
                stored.truncate(num_elements);
                stored
            }
            QuantStore::PackedU32(packed_dim) => match scheme.value {
                QuantValue::Q8F
                | QuantValue::Q8S
                | QuantValue::Q4F
                | QuantValue::Q4S
                | QuantValue::Q2F
                | QuantValue::Q2S => {
                    let words: Vec<u32> = bytemuck::cast_slice::<i8, u8>(&stored)
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|word| u32::from_ne_bytes(*word))
                        .collect();
                    unpack_u32(&words, &shape, packed_dim, scheme.value.size_bits())
                }
                QuantValue::E4M3 | QuantValue::E5M2 | QuantValue::E2M1 => {
                    unimplemented!("Not yet supported")
                }
            },
            QuantStore::PackedNative(_) => unimplemented!("Not yet supported"),
        };

        (values, (qparams, num_params))
    }
}

/// Round a scale up to the smallest value representable by the scale dtype that is no smaller.
///
/// Backends that keep scales in `f32` must apply this when quantizing, so that the scale they
/// divide by is the one that will actually be stored. Otherwise a tensor dequantizes differently
/// after a save/load round trip.
///
/// Up rather than to nearest, because a scale is derived from the largest magnitude it has to
/// cover. Rounding down puts that value past the end of the quantized range, where it clips, which
/// measured several times worse than the coarser step rounding up costs.
pub fn scale_to_dtype(scale: f32, dtype: ScaleDtype) -> f32 {
    dtype.round_up(scale)
}

/// Bytes taken by the per-tensor scale, zero for a scheme that does not carry one over blocks.
pub fn global_scale_size(scheme: &QuantScheme) -> usize {
    global_scale_dtype(scheme).map_or(0, scale_size)
}

/// Number of storage elements a tensor of `shape` occupies under `scheme`.
///
/// A packed store divides only the packed dimension, rounding that extent up on its own and
/// leaving the others intact, so a non-divisible extent pads once per line rather than once
/// over the flattened tensor. This mirrors the storage shape the allocation actually uses (see
/// `CubeTensor::quantized_storage` in burn-cubecl); flattening first would under-count, e.g. a
/// `[3, 3]` Q4 `PackedU32` tensor occupies `3 * ceil(3 / 8) = 3` words, not `ceil(9 / 8) = 2`.
fn storage_elements(scheme: &QuantScheme, shape: &Shape) -> usize {
    let num_quants = scheme.num_quants();

    match scheme.store {
        QuantStore::PackedU32(packed_dim) | QuantStore::PackedNative(packed_dim)
            if num_quants > 1 && !shape.is_empty() =>
        {
            let packed_dim = shape.num_dims() - packed_dim - 1;
            let mut storage = shape.clone();
            storage[packed_dim] = storage[packed_dim].div_ceil(num_quants);
            storage.num_elements()
        }
        _ => shape.num_elements().div_ceil(num_quants),
    }
}

/// Total bytes a tensor of `shape` occupies under `scheme`, laid out as [`QuantizedBytes::new`]
/// writes it: values, then block scales, then (for a two-level scheme) the per-tensor scale.
pub fn quantized_data_len(scheme: &QuantScheme, shape: &Shape) -> usize {
    let value_bytes = storage_elements(scheme, shape) * scheme.size_bits_stored().div_ceil(8);

    let num_params = params_shape(shape, scheme).num_elements();
    let scale_bytes = num_params * scale_size(scheme.scale_dtype());

    value_bytes + scale_bytes + global_scale_size(scheme)
}

/// Bytes per stored scale entry for the given scale dtype.
pub fn scale_size(dtype: ScaleDtype) -> usize {
    match dtype {
        ScaleDtype::F32 => 4,
        ScaleDtype::F16 | ScaleDtype::BF16 => 2,
        ScaleDtype::UE8M0 | ScaleDtype::UE4M3 => 1,
    }
}

/// Decode stored scale entries into f32.
fn decode_scales(bytes: &[u8], dtype: ScaleDtype) -> Vec<f32> {
    match dtype {
        ScaleDtype::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        ScaleDtype::F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| crate::f16::from_ne_bytes([c[0], c[1]]).to_f32())
            .collect(),
        ScaleDtype::BF16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| crate::bf16::from_ne_bytes([c[0], c[1]]).to_f32())
            .collect(),
        ScaleDtype::UE4M3 => bytes.iter().map(|b| e4m3::from_bits(*b).to_f32()).collect(),
        ScaleDtype::UE8M0 => bytes.iter().map(|b| ue8m0_to_f32(*b)).collect(),
    }
}

/// Encode f32 scales at the scale dtype for serialization.
fn encode_scales(scales: &[f32], dtype: ScaleDtype) -> Vec<u8> {
    match dtype {
        ScaleDtype::F32 => scales.iter().flat_map(|s| s.to_ne_bytes()).collect(),
        ScaleDtype::F16 => scales
            .iter()
            .flat_map(|s| crate::f16::from_f32(*s).to_ne_bytes())
            .collect(),
        ScaleDtype::BF16 => scales
            .iter()
            .flat_map(|s| crate::bf16::from_f32(*s).to_ne_bytes())
            .collect(),
        ScaleDtype::UE4M3 => scales
            .iter()
            .map(|s| e4m3::from_f32(*s).to_bits())
            .collect(),
        // `f32_to_ue8m0` rounds up, which is the rule for a scale and matches `scale_to_dtype`;
        // a scale reaching here has already been rounded onto the grid anyway.
        ScaleDtype::UE8M0 => scales.iter().map(|s| f32_to_ue8m0(*s)).collect(),
    }
}

fn read_bytes_to_i8(bytes: Bytes) -> Vec<i8> {
    match bytes.try_into_vec::<i8>() {
        Ok(val) => val,
        // Safety,
        //
        // `Vec<u8>` can be Re-interpreted as `Vec<i8>` since they share the same alignment.
        Err(bytes) => unsafe { core::mem::transmute::<Vec<u8>, Vec<i8>>(bytes.to_vec()) },
    }
}

/// `scheme` once its tensor's axes are permuted by `axes`: the block dims and the packed axis
/// move with the axes they belong to.
pub fn permuted_scheme(mut scheme: QuantScheme, rank: usize, axes: &[usize]) -> QuantScheme {
    scheme.permute_block_dims(rank, axes);

    if let QuantStore::PackedU32(packed_dim) | QuantStore::PackedNative(packed_dim) =
        &mut scheme.store
    {
        let packed_axis = rank - *packed_dim - 1;
        let new_axis = axes
            .iter()
            .position(|axis| *axis == packed_axis)
            .expect("Permute axes to contain the packed axis");
        *packed_dim = rank - new_axis - 1;
    }

    scheme
}

/// Where each line along `axis` starts in a row-major tensor of `shape`, in the order a packed
/// store keeps them: the tensor with `axis` swapped innermost, read row-major.
fn packed_line_starts(shape: &[usize], axis: usize) -> impl Iterator<Item = usize> + '_ {
    let rank = shape.len();
    let mut strides = vec![1; rank];
    for dim in (0..rank - 1).rev() {
        strides[dim] = strides[dim + 1] * shape[dim + 1];
    }
    let outer: Vec<usize> = (0..rank - 1)
        .map(|position| if position == axis { rank - 1 } else { position })
        .collect();
    let num_lines = outer.iter().map(|&dim| shape[dim]).product();

    (0..num_lines).map(move |mut line| {
        outer
            .iter()
            .rev()
            .map(|&dim| {
                let index = line % shape[dim];
                line /= shape[dim];
                index * strides[dim]
            })
            .sum()
    })
}

/// The stride and extent of the axis a store packs along, `packed_dim` counting from the
/// innermost; a scalar packs as one line of one value.
fn packed_axis(shape: &Shape, packed_dim: usize) -> (Vec<usize>, usize) {
    let dims = match shape.as_slice() {
        [] => vec![1],
        dims => dims.to_vec(),
    };
    let axis = dims.len() - packed_dim - 1;
    (dims, axis)
}

/// `values`, row-major over `shape`, packed into `u32` words of `bits`-wide values along the axis
/// `packed_dim` counts from the innermost, each line padded to whole words.
fn pack_u32(values: &[i8], shape: &Shape, packed_dim: usize, bits: usize) -> Vec<u32> {
    let (dims, axis) = packed_axis(shape, packed_dim);
    let (len, stride) = (dims[axis], dims[axis + 1..].iter().product::<usize>());
    let per_word = u32::BITS as usize / bits;
    let mask = (1u32 << bits) - 1;

    packed_line_starts(&dims, axis)
        .flat_map(|start| {
            (0..len.div_ceil(per_word)).map(move |word| {
                (0..per_word.min(len - word * per_word)).fold(0u32, |packed, k| {
                    let value = values[start + (word * per_word + k) * stride];
                    packed | (value as u32 & mask) << (k * bits)
                })
            })
        })
        .collect()
}

/// The values [`pack_u32`] packed, row-major over `shape`.
fn unpack_u32(words: &[u32], shape: &Shape, packed_dim: usize, bits: usize) -> Vec<i8> {
    let (dims, axis) = packed_axis(shape, packed_dim);
    let (len, stride) = (dims[axis], dims[axis + 1..].iter().product::<usize>());
    let words_per_line = len.div_ceil(u32::BITS as usize / bits);
    let value = match bits {
        8 => QuantValue::Q8S,
        4 => QuantValue::Q4S,
        2 => QuantValue::Q2S,
        _ => unreachable!("integer quantized values are 8, 4 or 2 bits"),
    };

    let mut values = vec![0; shape.num_elements()];
    if values.is_empty() {
        return values;
    }
    for (start, line) in packed_line_starts(&dims, axis).zip(words.chunks(words_per_line)) {
        for (k, unpacked) in unpack_q_to_i8s(line, len, &value).into_iter().enumerate() {
            values[start + k * stride] = unpacked;
        }
    }
    values
}

/// Pack signed 8-bit integer values into a sequence of unsigned 32-bit integers.
pub fn pack_i8s_to_u32s(values: Vec<i8>) -> Vec<u32> {
    // Shift and combine groups of four 8-bit values into a u32.
    // Same as doing this:
    //     let result = (d_u8 & 0xFF) << 24 | (c_u8 & 0xFF) << 16 | (b_u8 & 0xFF) << 8 | (a_u8 & 0xFF);
    #[cfg(target_endian = "big")]
    {
        values
            .chunks(4)
            .map(|x| {
                x.iter()
                    .enumerate()
                    .fold(0u32, |acc, (i, x)| acc | (*x as u32 & 0xFF) << (i * 8))
            })
            .collect()
    }

    // The order of bytes in little endian matches the above description, we just need to
    // handle padding when the number of values is not a factor of 4
    #[cfg(target_endian = "little")]
    {
        let mut values = values;
        let remainder = values.len() % 4;
        if remainder != 0 {
            // Pad with zeros
            values.extend(core::iter::repeat_n(0, 4 - remainder));
        }

        let len = values.len() / 4;
        let capacity = values.capacity() / 4;

        // Pre-forget the old vec and re-interpret as u32
        let mut values = core::mem::ManuallyDrop::new(values);
        let ptr = values.as_mut_ptr() as *mut u32;

        unsafe { Vec::from_raw_parts(ptr, len, capacity) }
    }
}

/// Unpack integer values into a sequence of signed 8-bit integers.
pub(crate) fn unpack_q_to_i8s<Q: PrimInt>(
    values: &[Q],
    numel: usize,
    value: &QuantValue,
) -> Vec<i8> {
    let size_store = size_of::<Q>() * 8;
    let size_quant = value.size_bits();
    let num_quants = size_store / size_quant;
    let mask = Q::from((1 << size_quant) - 1).unwrap();
    let sign_shift = 8 - size_quant; // sign extension for sub-byte values
    values
        .iter()
        .enumerate()
        .flat_map(|(i, &packed)| {
            // A single u32 could contain less than four 8-bit values...
            let n = core::cmp::min(num_quants, numel - i * num_quants);
            // Extract each 8-bit segment from u32 and cast back to i8
            // Same as doing this (when 4 values are fully packed):
            //     let a = (packed & 0xFF) as i8;
            //     let b = ((packed >> 8) & 0xFF) as i8;
            //     let c = ((packed >> 16) & 0xFF) as i8;
            //     let d = ((packed >> 24) & 0xFF) as i8;
            (0..n).map(move |i| {
                let raw = (packed >> (i * size_quant) & mask).to_u8().unwrap();
                ((raw << sign_shift) as i8) >> sign_shift
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {

    use super::*;
    use alloc::vec;

    /// Neither 3 nor 5 fills a word, so every packed line is padded.
    #[test]
    fn packed_values_round_trip_along_either_axis() {
        let shape = Shape::new([3, 5]);
        for value in [QuantValue::Q8S, QuantValue::Q4S, QuantValue::Q2S] {
            let (min, max) = value.range();
            let values: Vec<i8> = (0..15)
                .map(|i| (min as i32 + i * 7 % (max - min + 1.0) as i32) as i8)
                .collect();
            for packed_dim in [0, 1] {
                let scheme = QuantScheme::default()
                    .with_value(value)
                    .with_store(QuantStore::PackedU32(packed_dim));

                let bytes =
                    QuantizedBytes::new(values.clone(), shape.clone(), scheme, &[1.0], None);

                assert_eq!(bytes.bytes.len(), quantized_data_len(&scheme, &shape));
                assert_eq!(
                    bytes.into_vec_i8().0,
                    values,
                    "{value:?} along {packed_dim}"
                );
            }
        }
    }

    /// cubecl stores a tensor packed along an outer axis as its transpose packed innermost.
    #[test]
    fn packing_along_the_outer_axis_packs_the_transposed_rows() {
        let rows: [[i8; 8]; 2] = [[0, 1, 2, 3, 4, 5, 6, 7], [-1; 8]];
        let expected = [0x7654_3210u32, 0xFFFF_FFFF];
        let q4 = QuantScheme::default().with_value(QuantValue::Q4S);

        let inner = rows.concat();
        let packed = QuantizedBytes::new(
            inner,
            [2, 8],
            q4.with_store(QuantStore::PackedU32(0)),
            &[1.0],
            None,
        );
        let outer: Vec<i8> = (0..8).flat_map(|i| rows.map(|row| row[i])).collect();
        let packed_outer = QuantizedBytes::new(
            outer,
            [8, 2],
            q4.with_store(QuantStore::PackedU32(1)),
            &[1.0],
            None,
        );

        let words = |bytes: &QuantizedBytes| -> Vec<u32> {
            bytes.bytes[..8]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| u32::from_ne_bytes(*word))
                .collect()
        };
        assert_eq!(words(&packed), expected);
        assert_eq!(words(&packed_outer), expected);
    }

    #[test]
    fn should_pack_i8s_to_u32() {
        let packed = pack_i8s_to_u32s(vec![-128, 2, -3, 127]);

        assert_eq!(packed, vec![2147287680]);
    }

    #[test]
    fn should_pack_i8s_to_u32_padded() {
        let packed = pack_i8s_to_u32s(vec![-128, 2, -3, 127, 55]);
        let packed_padded = pack_i8s_to_u32s(vec![-128, 2, -3, 127, 55, 0, 0, 0]);

        assert_eq!(packed, vec![2147287680, 55]);
        assert_eq!(packed, packed_padded);
    }

    #[test]
    fn should_unpack_u32s_to_i8s() {
        let unpacked = unpack_q_to_i8s(&[2147287680u32], 4, &QuantValue::Q8S);

        assert_eq!(unpacked, vec![-128, 2, -3, 127]);
    }

    #[test]
    fn should_unpack_u32s_to_i8s_padded() {
        let unpacked = unpack_q_to_i8s(&[55u32], 1, &QuantValue::Q8S);

        assert_eq!(unpacked, vec![55]);
    }

    #[test]
    fn should_unpack_u32s_to_i8s_arange() {
        let unpacked = unpack_q_to_i8s(
            &[
                0u32, 286331136, 286331153, 572657937, 572662306, 857874978, 858993459, 858993459,
                1145324612, 1145324612, 1431655748, 1431655765, 1717982549, 1717986918, 2003199590,
                2004318071,
            ],
            128,
            &QuantValue::Q4S,
        );

        assert_eq!(
            unpacked,
            vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5,
                5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7
            ]
        );
    }

    #[test]
    fn should_pack_unpack_quantization_parameters_per_tensor_symmetric() {
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let scale = 0.03937008;
        let values = vec![0i8, 25, 51, 76, 102, 127];

        let q_bytes = QuantizedBytes::new(
            values.clone(),
            [2, 3],
            QuantScheme::default()
                .with_value(QuantValue::Q8S)
                .with_store(QuantStore::Native),
            &[scale],
            None,
        );

        let (q_values, qparams) = q_bytes.into_vec_i8();

        assert_eq!(qparams.block, vec![scale]);

        assert_eq!(q_values, values);
    }

    /// Backends divide by what `scale_to_dtype` returns and hand that same value to
    /// `encode_scales`. If encoding moved it, a tensor would dequantize differently after a
    /// save/load round trip, so the codec has to leave an already-rounded scale alone.
    #[test]
    fn scale_to_dtype_survives_the_codec() {
        // Includes values that saturate (500), land in e4m3's subnormals (1e-3), and underflow
        // it entirely (7.7e-4).
        let scales = [0.5f32, 0.3, 1.0 / 3.0, 500.0, 1e-3, 7.7e-4];

        for dtype in [
            ScaleDtype::F32,
            ScaleDtype::F16,
            ScaleDtype::BF16,
            ScaleDtype::UE4M3,
            ScaleDtype::UE8M0,
        ] {
            let rounded: Vec<f32> = scales.iter().map(|s| scale_to_dtype(*s, dtype)).collect();
            let via_codec = decode_scales(&encode_scales(&rounded, dtype), dtype);

            assert_eq!(
                rounded, via_codec,
                "the codec moves a scale {dtype:?} can already represent"
            );
            // 500 is past what e4m3 can hold, so it saturates rather than rounding up.
            for (scale, rounded) in scales.iter().zip(&rounded).filter(|(s, _)| **s < 500.0) {
                assert!(
                    rounded >= scale,
                    "{scale} rounded down to {rounded} for {dtype:?}"
                );
            }
        }
    }

    /// The two scale regions have different widths, and the length assertion pins the layout as
    /// dense: nothing is padded between them.
    #[test]
    fn should_pack_unpack_two_level_scales() {
        // Exactly representable, so this pins the layout rather than the formats' rounding.
        let block_scales = [0.5f32, 0.125];
        let global = 3.0f32;
        let values = vec![0i8, 25, 51, 76, 102, 127, -128, -1];

        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native)
            .per_block([4], ScaleDtype::UE4M3)
            .per_tensor(ScaleDtype::F32);

        let q_bytes = QuantizedBytes::new(values.clone(), [8], scheme, &block_scales, Some(global));

        // 8 values, one byte per UE4M3 block scale, a 4 byte f32 per-tensor scale.
        assert_eq!(q_bytes.bytes.len(), 8 + 2 + 4);

        let (q_values, scales) = q_bytes.into_vec_i8();

        assert_eq!(q_values, values);
        assert_eq!(scales.block, block_scales);
        assert_eq!(scales.global, Some(global));
    }

    /// MXFP4's scale level: one `ue8m0` byte per block, no per-tensor scale. Serialization has to
    /// carry it like any other, or a model in that format cannot be written or read back.
    ///
    /// The scales here are powers of two, which is every value `ue8m0` has — so the round trip is
    /// exact and a mismatch is a real defect rather than the format's own rounding.
    #[test]
    fn should_pack_unpack_ue8m0_block_scales() {
        let block_scales = [0.25f32, 8.0];
        let values = vec![0i8, 25, 51, 76, 102, 127, -128, -1];

        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native)
            .per_block([4], ScaleDtype::UE8M0);

        let q_bytes = QuantizedBytes::new(values.clone(), [8], scheme, &block_scales, None);

        // 8 values and one byte per block scale; no per-tensor scale rides along.
        assert_eq!(q_bytes.bytes.len(), 8 + 2);

        let (q_values, scales) = q_bytes.into_vec_i8();

        assert_eq!(q_values, values);
        assert_eq!(scales.block, block_scales);
        assert_eq!(scales.global, None);
    }

    #[test]
    #[should_panic(expected = "requires a per-tensor scale")]
    fn two_level_scheme_without_a_global_scale_is_rejected() {
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native)
            .per_block([4], ScaleDtype::F32)
            .per_tensor(ScaleDtype::F32);

        QuantizedBytes::new(vec![0i8; 8], [8], scheme, &[0.5, 0.125], None);
    }

    #[test]
    #[should_panic(expected = "stores its per-tensor scale as f32")]
    fn a_narrower_per_tensor_scale_is_rejected() {
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native)
            .per_block([4], ScaleDtype::UE4M3)
            .per_tensor(ScaleDtype::F16);

        QuantizedBytes::new(vec![0i8; 8], [8], scheme, &[0.5, 0.125], Some(3.0));
    }

    /// What a backend answers `supports_dtype` with, so a scheme it declines here is one no path
    /// reaches: each of these panics further in, where the rule is relied on.
    #[test]
    fn quantizable_declines_what_no_backend_can_store() {
        assert!(quantizable(&QuantScheme::default()));
        assert!(quantizable(
            &QuantScheme::default().per_block([4], ScaleDtype::F16)
        ));
        assert!(quantizable(
            &QuantScheme::default()
                .per_block([4], ScaleDtype::UE4M3)
                .per_tensor(ScaleDtype::F32)
        ));

        // `ue8m0` stores a bare exponent, so its round-up is to the next power of two.
        assert!(quantizable(
            &QuantScheme::default().per_block([4], ScaleDtype::UE8M0)
        ));
        assert!(quantizable(
            &QuantScheme::default().per_tensor(ScaleDtype::UE8M0)
        ));
        // MXFP4, whose whole definition is that pairing.
        assert!(quantizable(
            &QuantScheme::default()
                .with_value(QuantValue::E2M1)
                .per_block([32], ScaleDtype::UE8M0)
        ));

        // Block scales reaching f32's range leave the per-tensor scale subnormal, and a narrower
        // per-tensor scale rounds away precision every block was normalized against.
        assert!(!quantizable(
            &QuantScheme::default()
                .per_block([4], ScaleDtype::F32)
                .per_tensor(ScaleDtype::F32)
        ));
        assert!(!quantizable(
            &QuantScheme::default()
                .per_block([4], ScaleDtype::UE4M3)
                .per_tensor(ScaleDtype::BF16)
        ));
    }

    /// `scale_size` is what the readers use to locate the scales in the buffer, so an encoding
    /// wider or narrower than it claims silently misreads every scale.
    #[test]
    fn encoded_scale_width_matches_scale_size() {
        // Powers of two, so `ue8m0` carries them exactly like the wider dtypes do.
        let scales = [0.5f32, 0.25, 0.125];

        for dtype in [
            ScaleDtype::F32,
            ScaleDtype::F16,
            ScaleDtype::BF16,
            ScaleDtype::UE4M3,
            ScaleDtype::UE8M0,
        ] {
            assert_eq!(
                encode_scales(&scales, dtype).len(),
                scale_size(dtype) * scales.len(),
                "encoded width disagrees with scale_size for {dtype:?}"
            );
        }
    }

    #[test]
    fn should_pack_unpack_ue4m3_block_scales() {
        // Exactly representable in e4m3, so the round trip is lossless and the test pins the
        // layout rather than the format's rounding.
        let scales = [0.5f32, 0.125];
        let values = vec![0i8, 25, 51, 76, 102, 127, -128, -1];

        let q_bytes = QuantizedBytes::new(
            values.clone(),
            [8],
            QuantScheme::default()
                .with_value(QuantValue::Q8S)
                .with_store(QuantStore::Native)
                .per_block([4], ScaleDtype::UE4M3),
            &scales,
            None,
        );

        let (q_values, qparams) = q_bytes.into_vec_i8();

        assert_eq!(qparams.block, scales);
        assert_eq!(q_values, values);
    }
}
