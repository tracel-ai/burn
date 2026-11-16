use burn_std::Shape;
use smallvec::smallvec;

use crate::{OpSharding, Sharding};

/// Whether a reshape keeps a split: only when each member's chunk is still a contiguous chunk of
/// one output dim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReshapeRule {
    /// The input is not split, so each member reshapes it whole, a partial sum included.
    Local { sharding: Sharding },
    /// Every member's chunk of the split dim of `input` holds the same elements as its chunk of
    /// `to_dim`.
    Split { input: Sharding, to_dim: usize },
    /// The split dim is merged or broken up unevenly: the input is gathered first.
    Gathered,
}

impl ReshapeRule {
    pub fn new(input: Sharding, from: &Shape, to: &Shape, num_members: usize) -> Self {
        let (Some(dim), Some(chunks)) = (input.split_dim(), input.num_chunks(num_members)) else {
            return Self::Local { sharding: input };
        };
        match Self::chunk_preserving_dim(from, to, dim, chunks) {
            Some(to_dim) => Self::Split { input, to_dim },
            None => Self::Gathered,
        }
    }

    pub fn sharding(&self) -> OpSharding {
        let (input, output) = match *self {
            Self::Local { sharding } => (sharding, sharding),
            Self::Split { input, to_dim } => (input, input.on_dim(to_dim)),
            Self::Gathered => (Sharding::Replicated, Sharding::Replicated),
        };
        OpSharding {
            inputs: smallvec![input],
            output,
        }
    }

    /// The output dim starting where input dim `dim` starts in memory. Its chunks hold the same
    /// elements when it has the same length, or when both lengths split evenly into `chunks` and
    /// leave none empty.
    fn chunk_preserving_dim(from: &Shape, to: &Shape, dim: usize, chunks: usize) -> Option<usize> {
        let before = from[..dim].iter().product::<usize>();
        let len = from[dim];
        let mut prefix = 1;
        for (candidate, &size) in to.iter().enumerate() {
            if prefix == before && size != 1 {
                let same_chunks = size == len
                    || (len.is_multiple_of(chunks)
                        && size.is_multiple_of(chunks)
                        && size >= chunks);
                return same_chunks.then_some(candidate);
            }
            prefix *= size;
            if prefix > before {
                return None;
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heads_split_out_of_a_split_hidden_dim_stay_split() {
        let rule = ReshapeRule::new(
            Sharding::Split { dim: 2 },
            &Shape::new([2, 5, 12]),
            &Shape::new([2, 5, 4, 3]),
            2,
        );

        assert_eq!(
            rule,
            ReshapeRule::Split {
                input: Sharding::Split { dim: 2 },
                to_dim: 2
            }
        );
    }

    #[test]
    fn heads_merged_back_stay_split() {
        let rule = ReshapeRule::new(
            Sharding::Split { dim: 2 },
            &Shape::new([2, 5, 4, 3]),
            &Shape::new([2, 5, 12]),
            2,
        );

        assert_eq!(
            rule,
            ReshapeRule::Split {
                input: Sharding::Split { dim: 2 },
                to_dim: 2
            }
        );
    }

    #[test]
    fn heads_that_do_not_divide_over_the_members_are_gathered() {
        let rule = ReshapeRule::new(
            Sharding::Split { dim: 2 },
            &Shape::new([2, 5, 12]),
            &Shape::new([2, 5, 3, 4]),
            2,
        );

        assert_eq!(rule, ReshapeRule::Gathered);
    }

    #[test]
    fn flattening_a_column_split_is_gathered() {
        let rule = ReshapeRule::new(
            Sharding::Split { dim: 1 },
            &Shape::new([4, 6]),
            &Shape::new([24]),
            2,
        );

        assert_eq!(rule, ReshapeRule::Gathered);
    }

    #[test]
    fn an_unsqueezed_bias_keeps_its_split() {
        let rule = ReshapeRule::new(
            Sharding::Split { dim: 0 },
            &Shape::new([5]),
            &Shape::new([1, 5]),
            3,
        );

        assert_eq!(
            rule,
            ReshapeRule::Split {
                input: Sharding::Split { dim: 0 },
                to_dim: 1
            }
        );
    }

    #[test]
    fn folded_batch_dims_keep_a_split_last_dim() {
        let rule = ReshapeRule::new(
            Sharding::Split { dim: 2 },
            &Shape::new([2, 4, 12]),
            &Shape::new([8, 12]),
            2,
        );

        assert_eq!(
            rule,
            ReshapeRule::Split {
                input: Sharding::Split { dim: 2 },
                to_dim: 1
            }
        );
    }

    #[test]
    fn key_heads_shared_by_pairs_split_out_of_the_projection_keep_their_groups() {
        let kv = Sharding::ReplicatedSplit { dim: 2, groups: 2 };

        let rule = ReshapeRule::new(kv, &Shape::new([1, 5, 16]), &Shape::new([1, 5, 2, 8]), 4);

        assert_eq!(rule.sharding().output, kv);
    }
}
