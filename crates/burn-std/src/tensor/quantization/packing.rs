use alloc::{vec, vec::Vec};

use super::{QuantScheme, QuantStore, QuantValue, QuantValueCodes, params_shape};
use crate::Shape;

/// How [`QuantizedBytes`](super::QuantizedBytes) lays out the values of a scheme.
#[derive(Clone, Copy)]
pub enum ValueLayout {
    /// One byte per value.
    Bytes,
    /// Values packed into words along the axis `packed_dim` counts from the innermost, as
    /// [`PackedOrder`] orders them.
    Packed {
        packed_dim: usize,
        word: PackedWordLayout,
    },
}

impl ValueLayout {
    pub fn new(scheme: &QuantScheme) -> Self {
        match scheme.store {
            QuantStore::Native => Self::Bytes,
            QuantStore::PackedU32(packed_dim) | QuantStore::PackedNative(packed_dim) => {
                Self::Packed {
                    packed_dim,
                    word: PackedWordLayout::new(scheme),
                }
            }
        }
    }

    /// The order the block scales of a tensor of `shape` are stored in under `scheme`, when that
    /// is not row-major: blocks of values packed along an outer axis.
    pub fn block_scale_order(&self, shape: &Shape, scheme: &QuantScheme) -> Option<PackedOrder> {
        match *self {
            Self::Packed { packed_dim, .. } if packed_dim != 0 && scheme.block_size().is_some() => {
                Some(PackedOrder::new(
                    params_shape(shape, scheme).as_slice(),
                    packed_dim,
                ))
            }
            _ => None,
        }
    }
}

/// How values sit side by side in a little-endian word, value `k` at bit `k * bits`: a `u32` for
/// `PackedU32`, the native type for `PackedNative`, such as a byte of two E2M1 codes.
#[derive(Clone, Copy)]
pub struct PackedWordLayout {
    bytes: usize,
    values: usize,
    value: QuantValue,
}

impl PackedWordLayout {
    fn new(scheme: &QuantScheme) -> Self {
        Self {
            bytes: scheme.size_bits_stored().div_ceil(u8::BITS as usize),
            values: scheme.num_quants(),
            value: scheme.value,
        }
    }

    fn bits(&self) -> usize {
        self.value.size_bits()
    }

    /// The code a field holds: an integer narrower than a byte is sign-extended, a float code
    /// kept as its bits.
    fn code(&self, field: u8) -> i8 {
        if self.value.codes_are_integers() {
            let shift = u8::BITS as usize - self.bits();
            ((field << shift) as i8) >> shift
        } else {
            field as i8
        }
    }
}

/// A tensor of `dims` as cubecl stores it when packed along an axis: with that axis swapped
/// innermost, read row-major. Its values and its block scales are both laid out this way.
pub struct PackedOrder {
    dims: Vec<usize>,
    axis: usize,
}

impl PackedOrder {
    /// The order a store packed along `packed_dim`, counted from the innermost axis, gives a
    /// tensor of `dims`; a scalar is one line of one value.
    pub fn new(dims: &[usize], packed_dim: usize) -> Self {
        let dims = match dims {
            [] => vec![1],
            dims => dims.to_vec(),
        };
        assert!(
            packed_dim < dims.len(),
            "a store packed along dim {packed_dim} from the innermost needs more than {} dims",
            dims.len()
        );
        let axis = dims.len() - packed_dim - 1;
        Self { dims, axis }
    }

    /// Each line along the packed axis, in stored order, as the row-major positions of its values.
    fn lines(&self) -> impl Iterator<Item = impl Iterator<Item = usize>> + '_ {
        let rank = self.dims.len();
        let stride_of = |dim: usize| self.dims[dim + 1..].iter().product::<usize>();
        let (len, stride) = (self.dims[self.axis], stride_of(self.axis));
        let outer: Vec<usize> = (0..rank - 1)
            .map(|position| {
                if position == self.axis {
                    rank - 1
                } else {
                    position
                }
            })
            .collect();
        let num_lines = outer.iter().map(|&dim| self.dims[dim]).product();

        (0..num_lines).map(move |mut line| {
            let start: usize = outer
                .iter()
                .rev()
                .map(|&dim| {
                    let index = line % self.dims[dim];
                    line /= self.dims[dim];
                    index * stride_of(dim)
                })
                .sum();
            (0..len).map(move |k| start + k * stride)
        })
    }

    /// Row-major `values` as this order stores them.
    pub fn to_stored<T: Copy>(&self, values: &[T]) -> Vec<T> {
        self.lines()
            .flatten()
            .map(|position| values[position])
            .collect()
    }

    /// Stored `values` back in row-major order.
    pub fn to_row_major<T: Copy + Default>(&self, stored: &[T]) -> Vec<T> {
        let mut values = vec![T::default(); stored.len()];
        for (position, value) in self.lines().flatten().zip(stored) {
            values[position] = *value;
        }
        values
    }

    /// The values along the packed axis.
    fn line_len(&self) -> usize {
        self.dims[self.axis]
    }

    /// The words a line takes, its last one padded.
    fn words_per_line(&self, word: &PackedWordLayout) -> usize {
        self.line_len().div_ceil(word.values)
    }

    /// Whether packing copies each row as it is: a byte per value along the innermost axis, which
    /// little-endian words hold in order.
    fn copies_rows(&self, word: &PackedWordLayout) -> bool {
        self.axis == self.dims.len() - 1 && word.bits() == u8::BITS as usize && self.line_len() > 0
    }

    /// Row-major `values` packed into words, every line padded to whole words.
    pub fn pack(&self, values: &[i8], word: &PackedWordLayout) -> Vec<u8> {
        let words_per_line = self.words_per_line(word);
        if self.copies_rows(word) {
            let padding = words_per_line * word.bytes - self.line_len();
            return bytemuck::cast_slice::<i8, u8>(values)
                .chunks(self.line_len())
                .flat_map(|row| row.iter().copied().chain(core::iter::repeat_n(0, padding)))
                .collect();
        }
        let bits = word.bits();
        let mask = (1u32 << bits) - 1;
        let mut words = Vec::new();
        for line in self.lines() {
            let start = words.len();
            words.resize(start + words_per_line, 0u32);
            for (k, position) in line.enumerate() {
                words[start + k / word.values] |=
                    (values[position] as u32 & mask) << (k % word.values * bits);
            }
        }
        words
            .into_iter()
            .flat_map(|packed| packed.to_le_bytes().into_iter().take(word.bytes))
            .collect()
    }

    /// The row-major values [`pack`](Self::pack) packed into `bytes`.
    pub fn unpack(&self, bytes: &[u8], word: &PackedWordLayout) -> Vec<i8> {
        let (len, words_per_line) = (self.line_len(), self.words_per_line(word));
        if self.copies_rows(word) {
            return bytes
                .chunks(words_per_line * word.bytes)
                .flat_map(|row| bytemuck::cast_slice::<u8, i8>(&row[..len]).iter().copied())
                .collect();
        }
        let words: Vec<u32> = bytes
            .chunks(word.bytes)
            .map(|packed| {
                let mut le = [0u8; size_of::<u32>()];
                le[..packed.len()].copy_from_slice(packed);
                u32::from_le_bytes(le)
            })
            .collect();
        let bits = word.bits();
        let mask = (1u32 << bits) - 1;
        let mut values = vec![0; self.dims.iter().product()];
        for (index, line) in self.lines().enumerate() {
            let line_words = &words[index * words_per_line..][..words_per_line];
            for (k, position) in line.enumerate() {
                let field = line_words[k / word.values] >> (k % word.values * bits) & mask;
                values[position] = word.code(field as u8);
            }
        }
        values
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytes::Bytes;
    use crate::tensor::quantization::{QuantizedBytes, ScaleDtype, quantized_data_len};
    use alloc::vec;

    /// The first `count` packed words of `bytes`.
    fn words(bytes: &QuantizedBytes, count: usize) -> Vec<u32> {
        bytes.bytes[..count * size_of::<u32>()]
            .as_chunks::<{ size_of::<u32>() }>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect()
    }

    /// No packed extent here is a whole number of words, so every line is padded.
    #[test]
    fn packed_values_round_trip_along_any_axis() {
        for shape in [Shape::new([3, 5]), Shape::new([2, 3, 5])] {
            for value in [QuantValue::Q8S, QuantValue::Q4S, QuantValue::Q2S] {
                let (min, max) = value.range();
                let values: Vec<i8> = (0..shape.num_elements() as i32)
                    .map(|i| (min as i32 + i * 7 % (max - min + 1.0) as i32) as i8)
                    .collect();
                for packed_dim in 0..shape.num_dims() {
                    let scheme = QuantScheme::default()
                        .with_value(value)
                        .with_store(QuantStore::PackedU32(packed_dim));

                    let bytes =
                        QuantizedBytes::new(values.clone(), shape.clone(), scheme, &[1.0], None);

                    assert_eq!(bytes.bytes.len(), quantized_data_len(&scheme, &shape));
                    assert_eq!(
                        bytes.into_vec_i8().0,
                        values,
                        "{value:?} along {packed_dim} of {shape:?}"
                    );
                }
            }
        }
    }

    /// cubecl keeps the block scales of a tensor packed along an outer axis in the same swapped
    /// order as its values.
    #[test]
    fn block_scales_follow_the_values_when_packed_along_an_outer_axis() {
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::PackedU32(1))
            .per_block([2, 4], ScaleDtype::F32);
        let scales = [1.0f32, 2.0, 3.0, 4.0];

        let bytes = QuantizedBytes::new(vec![0i8; 32], [4, 8], scheme, &scales, None);

        let stored: Vec<f32> = bytes.bytes[bytes.bytes.len() - scales.len() * size_of::<f32>()..]
            .as_chunks::<{ size_of::<f32>() }>()
            .0
            .iter()
            .map(|scale| f32::from_ne_bytes(*scale))
            .collect();
        assert_eq!(stored, [1.0, 3.0, 2.0, 4.0]);
        assert_eq!(bytes.into_vec_i8().1.block, scales);
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

        assert_eq!(words(&packed, 2), expected);
        assert_eq!(words(&packed_outer, 2), expected);
    }

    /// At rank 3, swapping the packed axis innermost and moving it there store different words.
    #[test]
    fn packing_along_an_outer_axis_swaps_it_innermost() {
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::PackedU32(2));

        let packed = QuantizedBytes::new((0..24i8).collect(), [4, 2, 3], scheme, &[1.0], None);

        assert_eq!(
            words(&packed, 6),
            [
                0x120C_0600,
                0x150F_0903,
                0x130D_0701,
                0x1610_0A04,
                0x140E_0802,
                0x1711_0B05
            ]
        );
    }

    /// One-byte float codes are laid out as Q8 values are.
    #[test]
    fn packing_bytes_along_rows_pads_each_row() {
        for value in [QuantValue::Q8S, QuantValue::E4M3] {
            let scheme = QuantScheme::default()
                .with_value(value)
                .with_store(QuantStore::PackedU32(0));

            let packed = QuantizedBytes::new((0..6i8).collect(), [2, 3], scheme, &[1.0], None);

            assert_eq!(words(&packed, 2), [0x0002_0100, 0x0005_0403], "{value:?}");
            assert_eq!(packed.into_vec_i8().0, (0..6i8).collect::<Vec<_>>());
        }
    }

    #[test]
    #[should_panic(expected = "take 12 bytes, not 9")]
    fn values_whose_lines_are_not_padded_are_refused() {
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::PackedU32(0));
        let mut bytes = vec![0u8; 9];
        bytes.extend(1.0f32.to_ne_bytes());

        QuantizedBytes {
            bytes: Bytes::from_elems(bytes),
            scheme,
            shape: Shape::new([3, 3]),
        }
        .into_vec_i8();
    }

    #[test]
    fn float_codes_round_trip_along_any_axis_in_any_packed_store() {
        for shape in [Shape::new([3, 5]), Shape::new([2, 3, 5])] {
            for (value, codes) in [(QuantValue::E2M1, 16), (QuantValue::E4M3, 256)] {
                let values: Vec<i8> = (0..shape.num_elements())
                    .map(|i| (i * 7 % codes) as u8 as i8)
                    .collect();
                for packed_dim in 0..shape.num_dims() {
                    for store in [
                        QuantStore::PackedU32(packed_dim),
                        QuantStore::PackedNative(packed_dim),
                    ] {
                        let scheme = QuantScheme::default().with_value(value).with_store(store);

                        let bytes = QuantizedBytes::new(
                            values.clone(),
                            shape.clone(),
                            scheme,
                            &[1.0],
                            None,
                        );

                        assert_eq!(bytes.bytes.len(), quantized_data_len(&scheme, &shape));
                        assert_eq!(bytes.into_vec_i8().0, values, "{value:?} in {store:?}");
                    }
                }
            }
        }
    }

    /// cubecl's native E2M1 type holds two codes to a byte, the first in the low nibble.
    #[test]
    fn e2m1_packs_two_codes_to_a_byte_natively() {
        let scheme = QuantScheme::default()
            .with_value(QuantValue::E2M1)
            .with_store(QuantStore::PackedNative(0));

        let packed = QuantizedBytes::new(vec![0x1i8, 0xF, 0x6], [3], scheme, &[1.0], None);

        assert_eq!(&packed.bytes[..2], [0xF1, 0x06]);
    }
}
