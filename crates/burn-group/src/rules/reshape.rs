use burn_std::Shape;

use crate::{OpPlacement, Placement};

/// Whether a reshape keeps a split: only when each rank's chunk is still a contiguous chunk of
/// one output dim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReshapeRule {
    /// The input is not split: each rank reshapes it whole.
    Local { placement: Placement },
    /// Every rank's chunk of `from_dim` holds the same elements as its chunk of `to_dim`.
    Sharded { from_dim: usize, to_dim: usize },
    /// The split dim is merged or broken up unevenly: the input is gathered first.
    Gathered,
}

impl ReshapeRule {
    pub fn new(input: Placement, from: &Shape, to: &Shape, ranks: usize) -> Self {
        let Placement::Sharded { dim } = input else {
            return Self::Local { placement: input };
        };
        match Self::chunk_preserving_dim(from, to, dim, ranks) {
            Some(to_dim) => Self::Sharded {
                from_dim: dim,
                to_dim,
            },
            None => Self::Gathered,
        }
    }

    pub fn placement(&self) -> OpPlacement<1> {
        let (input, output) = match *self {
            Self::Local { placement } => (placement, placement),
            Self::Sharded { from_dim, to_dim } => (
                Placement::Sharded { dim: from_dim },
                Placement::Sharded { dim: to_dim },
            ),
            Self::Gathered => (Placement::Replicated, Placement::Replicated),
        };
        OpPlacement {
            inputs: [input],
            output,
        }
    }

    /// The output dim starting where input dim `dim` starts in memory. Its chunks hold the same
    /// elements when it has the same length, or when both lengths split evenly over the ranks.
    fn chunk_preserving_dim(from: &Shape, to: &Shape, dim: usize, ranks: usize) -> Option<usize> {
        let before = from[..dim].iter().product::<usize>();
        let len = from[dim];
        let mut prefix = 1;
        for (candidate, &size) in to.iter().enumerate() {
            if prefix == before && size != 1 {
                let same_chunks =
                    size == len || (len.is_multiple_of(ranks) && size.is_multiple_of(ranks));
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
    fn heads_split_out_of_a_sharded_hidden_dim_stay_sharded() {
        let rule = ReshapeRule::new(
            Placement::Sharded { dim: 2 },
            &Shape::new([2, 5, 12]),
            &Shape::new([2, 5, 4, 3]),
            2,
        );
        assert_eq!(
            rule,
            ReshapeRule::Sharded {
                from_dim: 2,
                to_dim: 2
            }
        );
    }

    #[test]
    fn heads_merged_back_stay_sharded() {
        let rule = ReshapeRule::new(
            Placement::Sharded { dim: 2 },
            &Shape::new([2, 5, 4, 3]),
            &Shape::new([2, 5, 12]),
            2,
        );
        assert_eq!(
            rule,
            ReshapeRule::Sharded {
                from_dim: 2,
                to_dim: 2
            }
        );
    }

    #[test]
    fn heads_that_do_not_divide_over_the_ranks_are_gathered() {
        let rule = ReshapeRule::new(
            Placement::Sharded { dim: 2 },
            &Shape::new([2, 5, 12]),
            &Shape::new([2, 5, 3, 4]),
            2,
        );
        assert_eq!(rule, ReshapeRule::Gathered);
    }

    #[test]
    fn flattening_a_column_split_is_gathered() {
        let rule = ReshapeRule::new(
            Placement::Sharded { dim: 1 },
            &Shape::new([4, 6]),
            &Shape::new([24]),
            2,
        );
        assert_eq!(rule, ReshapeRule::Gathered);
    }

    #[test]
    fn unsqueezed_bias_keeps_its_split() {
        let rule = ReshapeRule::new(
            Placement::Sharded { dim: 0 },
            &Shape::new([5]),
            &Shape::new([1, 5]),
            3,
        );
        assert_eq!(
            rule,
            ReshapeRule::Sharded {
                from_dim: 0,
                to_dim: 1
            }
        );
    }
}
