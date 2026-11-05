use burn_tensor::kind::Basic;
use burn_tensor::quantization::ScaleDtype;
use burn_tensor::{DType, Tensor};

/// What one component must move and compute to run **alone**, at best: with nothing of its own
/// reaching global memory twice.
///
/// A two-wall roofline. One wall is traffic through global memory, declared per direction as
/// [`read`](Self::read) and [`write`](Self::write); the other is [`flops`](Self::flops), the
/// inner loop's logical arithmetic. A component's floor is the taller of the two quotients
/// against the device's peaks. Neither says anything about shared memory, registers or tensor
/// cores.
///
/// These are **lower bounds**: no execution moves fewer bytes or does less arithmetic, and one
/// that appears to has been mispriced. They describe the component in isolation, so fusing it
/// into a neighbour goes below them. A composite's own bounds therefore sit below the sum of its
/// children's in each direction ([`moves_within`](Self::moves_within)), and the gap is the
/// intermediate traffic a perfect fuser would remove.
///
/// A floor that depends on the values, such as the distinct rows a lookup selects, may read them
/// back: the `Instrument` evaluates bounds in a pricing run of their own, and
/// the run it times evaluates none. Leave bounds off where a component will be fused into its
/// neighbour: a floor no achievable kernel can meet is worse than none.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "std", serde(deny_unknown_fields))]
pub struct Bounds {
    /// Bytes read from global memory, at the widths the tensors are stored in (see
    /// [`tensor_bytes`]).
    ///
    /// Declared apart from [`write`](Self::write) because a device's read and write ceilings
    /// differ, and a single total cannot be scored against either: a weight-streaming matmul
    /// that writes one short vector is almost pure read and can run above the device's copy
    /// ceiling.
    pub read: u64,
    /// Bytes written to global memory. See [`read`](Self::read).
    pub write: u64,
    /// Multiply-add FLOPs, at 2 per multiply-add.
    ///
    /// Logical only: dequantizing a quantized operand is real work and absent here, so a
    /// quantized component's achieved compute is not comparable to a float run of the same
    /// shape.
    pub flops: u64,
}

impl Bounds {
    /// Everything the component moves, both directions.
    ///
    /// The figure to print or to compare two declarations by, not the one to divide by a peak,
    /// since each direction has its own.
    pub fn bytes(self) -> u64 {
        self.read + self.write
    }

    /// Whether this moves no more than `ceiling` in **either** direction: the relation a
    /// composite's own bounds hold against the sum of its parts'.
    ///
    /// Per direction rather than over [`bytes`](Self::bytes): a composite that shifted traffic
    /// from reads to writes inside one unchanged total would declare a floor above its parts'.
    pub fn moves_within(self, ceiling: Bounds) -> bool {
        self.read <= ceiling.read && self.write <= ceiling.write
    }
}

impl core::ops::Add for Bounds {
    type Output = Bounds;

    fn add(self, other: Bounds) -> Bounds {
        Bounds {
            read: self.read + other.read,
            write: self.write + other.write,
            flops: self.flops + other.flops,
        }
    }
}

impl core::iter::Sum for Bounds {
    fn sum<I: Iterator<Item = Bounds>>(iter: I) -> Bounds {
        iter.fold(Bounds::default(), |total, bounds| total + bounds)
    }
}

/// The bytes a tensor occupies **as it is stored**: for a quantized tensor, its packed values
/// plus its scales, not the width its elements read back as. Any kind of tensor: indices and
/// masks are traffic too.
///
/// Why [`Bounds`] are declared by the component rather than computed for it: the component holds
/// the tensor and so knows its scheme, and no caller has to restate it.
pub fn tensor_bytes<const D: usize, K: Basic>(tensor: &Tensor<D, K>) -> u64 {
    let elements: u64 = tensor.dims().iter().map(|dim| *dim as u64).product();
    match tensor.dtype() {
        DType::QFloat(scheme) => {
            let values = elements * scheme.size_bits_value() as u64 / 8;
            // A two-level scheme adds one per-tensor scalar on top of the block scales, which is
            // below this model's resolution.
            let scales = match scheme.block_scale() {
                None => 1,
                Some(block) => elements / block.size.num_elements() as u64,
            };
            let scale_bytes = match scheme.scale_dtype() {
                ScaleDtype::F32 => 4,
                ScaleDtype::F16 | ScaleDtype::BF16 => 2,
                ScaleDtype::UE8M0 | ScaleDtype::UE4M3 => 1,
            };
            values + scales * scale_bytes
        }
        dtype => elements * dtype.size() as u64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An index or mask tensor is priced as stored too: its elements at their width.
    #[test]
    fn an_index_tensor_occupies_its_elements_at_their_width() {
        let device = Default::default();
        let indices = Tensor::<2, burn_tensor::Int>::zeros([3, 4], &device);
        let mask = indices.clone().equal_elem(1);

        assert_eq!(tensor_bytes(&indices), 12 * indices.dtype().size() as u64);
        assert_eq!(tensor_bytes(&mask), 12 * mask.dtype().size() as u64);
    }

    /// A declaration that moved traffic from reads to writes inside one unchanged total is
    /// rejected; held on the total, it would pass with a floor above its parts'.
    #[test]
    fn shifting_traffic_between_directions_does_not_move_within() {
        let parts = Bounds {
            read: 100,
            write: 20,
            flops: 0,
        };
        let shifted = Bounds {
            read: 20,
            write: 100,
            flops: 0,
        };
        assert_eq!(shifted.bytes(), parts.bytes(), "the total is untouched");
        assert!(!shifted.moves_within(parts));

        let fused = Bounds {
            read: 90,
            write: 15,
            flops: 0,
        };
        assert!(fused.moves_within(parts));
        assert!(parts.moves_within(parts), "the relation is not strict");
    }

    /// Bounds sum per direction, which is what lets a composite's parts be compared with it read
    /// for read and write for write.
    #[test]
    fn bounds_add_per_direction() {
        let parts = [
            Bounds {
                read: 8,
                write: 2,
                flops: 4,
            },
            Bounds {
                read: 5,
                write: 1,
                flops: 2,
            },
        ];
        let sum: Bounds = parts.into_iter().sum();
        assert_eq!(
            sum,
            Bounds {
                read: 13,
                write: 3,
                flops: 6
            }
        );
    }
}
