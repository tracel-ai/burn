use burn_std::{Shape, Slice};
use smallvec::smallvec;

use crate::{
    OpSharding,
    Sharding::{self, PartialSum, Replicated, Split},
};

/// An op that only moves dims or repeats along them: a split follows its dim, and a whole value
/// or a partial sum stays one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayoutRule {
    input: Sharding,
    output: Sharding,
}

impl LayoutRule {
    /// Output dim `i` is input dim `axes[i]`.
    pub fn permute(input: Sharding, axes: &[usize]) -> Self {
        let output = match input {
            Split { dim } => Split {
                dim: axes
                    .iter()
                    .position(|&axis| axis == dim)
                    .expect("A permutation holds every dim"),
            },
            sharding => sharding,
        };
        Self { input, output }
    }

    pub fn swap(input: Sharding, dim1: usize, dim2: usize) -> Self {
        let output = match input {
            Split { dim } if dim == dim1 => Split { dim: dim2 },
            Split { dim } if dim == dim2 => Split { dim: dim1 },
            sharding => sharding,
        };
        Self { input, output }
    }

    /// Expanding prepends the dims the output has beyond the input's, and repeats only dims of
    /// length 1, which a split never has.
    pub fn expand(input: Sharding, input_dims: usize, output_dims: usize) -> Self {
        let output = match input {
            Split { dim } => Split {
                dim: dim + output_dims - input_dims,
            },
            sharding => sharding,
        };
        Self { input, output }
    }

    /// Slicing keeps a split only when it takes the split dim whole: each member then slices its
    /// own shard along the other dims.
    pub fn slice(input: Sharding, ranges: &[Slice], shape: &Shape) -> Self {
        let sharding = match input {
            Split { dim } if !Self::takes_whole(ranges.get(dim), shape[dim]) => Replicated,
            sharding => sharding,
        };
        Self {
            input: sharding,
            output: sharding,
        }
    }

    /// Assigning into a slice keeps the first split of the tensor or the values whose dim the slice
    /// takes whole, then splits the other one alike. Both partial stays partial, since the
    /// assignment is linear in the two together.
    pub fn slice_assign(
        tensor: Sharding,
        value: Sharding,
        ranges: &[Slice],
        shape: &Shape,
    ) -> Self {
        let split = [tensor, value].into_iter().find(|sharding| match sharding {
            Split { dim } => Self::takes_whole(ranges.get(*dim), shape[*dim]),
            _ => false,
        });
        let sharding = match (split, tensor, value) {
            (Some(split), _, _) => split,
            (None, PartialSum, PartialSum) => PartialSum,
            _ => Replicated,
        };
        Self {
            input: sharding,
            output: sharding,
        }
    }

    /// Concatenating along `dim` keeps a split of another dim, which every input then takes, each
    /// member laying its own shards end to end. Partial sums all stay one, the concatenation
    /// being linear in its inputs together; anything else, a split of `dim` included, gathers.
    pub fn concat(inputs: &[Sharding], dim: usize) -> Self {
        let split = inputs
            .iter()
            .copied()
            .find(|sharding| matches!(sharding, Split { .. }));
        let compatible = |split: Sharding| {
            inputs
                .iter()
                .all(|sharding| *sharding == split || *sharding == Replicated)
        };
        let sharding = match split {
            Some(Split { dim: split_dim })
                if split_dim != dim && compatible(Split { dim: split_dim }) =>
            {
                Split { dim: split_dim }
            }
            Some(_) => Replicated,
            None if inputs.iter().all(|sharding| *sharding == PartialSum) => PartialSum,
            None => Replicated,
        };
        Self {
            input: sharding,
            output: sharding,
        }
    }

    fn takes_whole(range: Option<&Slice>, len: usize) -> bool {
        range.is_none_or(|range| range.step == 1 && range.to_range(len) == (0..len))
    }

    /// Every one of the op's `num_inputs` shares the one sharding.
    pub fn sharding(&self, num_inputs: usize) -> OpSharding {
        OpSharding {
            inputs: smallvec![self.input; num_inputs],
            output: self.output,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_concat_along_another_dim_keeps_the_split_and_splits_whole_inputs_alike() {
        let rule = LayoutRule::concat(&[Split { dim: 1 }, Replicated], 3);

        assert_eq!(rule.sharding(2).inputs.as_slice(), [Split { dim: 1 }; 2]);
        assert_eq!(rule.sharding(2).output, Split { dim: 1 });
    }

    #[test]
    fn a_concat_along_the_split_dim_or_of_two_splits_gathers() {
        for inputs in [
            [Split { dim: 3 }, Split { dim: 3 }],
            [Split { dim: 1 }, Split { dim: 2 }],
            [Split { dim: 1 }, PartialSum],
        ] {
            assert_eq!(
                LayoutRule::concat(&inputs, 3).sharding(2).output,
                Replicated,
                "{inputs:?}"
            );
        }
    }

    #[test]
    fn a_concat_of_partial_sums_stays_partial() {
        let rule = LayoutRule::concat(&[PartialSum, PartialSum], 0);

        assert_eq!(rule.sharding(2).output, PartialSum);
    }

    #[test]
    fn heads_split_moves_with_its_dim_when_heads_go_outward() {
        let rule = LayoutRule::swap(Split { dim: 2 }, 1, 2);

        assert_eq!(rule.sharding(1).output, Split { dim: 1 });
    }

    #[test]
    fn a_split_follows_its_dim_through_a_permutation() {
        let rule = LayoutRule::permute(Split { dim: 0 }, &[2, 0, 1]);

        assert_eq!(rule.sharding(1).output, Split { dim: 1 });
    }

    #[test]
    fn a_split_stays_on_its_dim_when_expand_prepends_dims() {
        let rule = LayoutRule::expand(Split { dim: 1 }, 4, 5);

        assert_eq!(rule.sharding(1).output, Split { dim: 2 });
    }

    #[test]
    fn a_slice_of_another_dim_keeps_the_split() {
        let shape = Shape::new([2, 4, 3, 8]);
        let ranges = [0..2, 0..4, 0..3, 0..4].map(Slice::from);

        let rule = LayoutRule::slice(Split { dim: 1 }, &ranges, &shape);

        assert_eq!(rule.sharding(1).output, Split { dim: 1 });
    }

    #[test]
    fn a_slice_cutting_the_split_dim_gathers() {
        let shape = Shape::new([2, 4]);
        let ranges = [0..2, 1..3].map(Slice::from);

        let rule = LayoutRule::slice(Split { dim: 1 }, &ranges, &shape);

        assert_eq!(rule.sharding(1).inputs[..], [Replicated]);
    }

    #[test]
    fn rotated_halves_assigned_into_a_split_tensor_keep_its_split() {
        let shape = Shape::new([1, 4, 3, 8]);
        let ranges = [0..1, 0..4, 0..3, 4..8].map(Slice::from);

        let rule = LayoutRule::slice_assign(Replicated, Split { dim: 1 }, &ranges, &shape);

        assert_eq!(
            rule.sharding(2),
            OpSharding {
                inputs: smallvec![Split { dim: 1 }; 2],
                output: Split { dim: 1 },
            }
        );
    }

    #[test]
    fn a_partial_sum_stays_partial() {
        let rule = LayoutRule::swap(PartialSum, 0, 1);

        assert_eq!(rule.sharding(1).output, PartialSum);
    }
}
