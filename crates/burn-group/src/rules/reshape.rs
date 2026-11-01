use burn_std::Shape;

use crate::{OpSharding, Sharding};

/// Whether a reshape keeps a split: only when each member's chunk is still a contiguous chunk of
/// one output dim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReshapeRule {
    /// The input is not split, so each member reshapes it whole, a partial sum included.
    Local { sharding: Sharding },
    /// Every member's chunk of `from_dim` holds the same elements as its chunk of `to_dim`.
    Split { from_dim: usize, to_dim: usize },
    /// The split dim is merged or broken up unevenly: the input is gathered first.
    Gathered,
}

impl ReshapeRule {
    pub fn new(input: Sharding, from: &Shape, to: &Shape, num_members: usize) -> Self {
        let Sharding::Split { dim } = input else {
            return Self::Local { sharding: input };
        };
        match Self::chunk_preserving_dim(from, to, dim, num_members) {
            Some(to_dim) => Self::Split {
                from_dim: dim,
                to_dim,
            },
            None => Self::Gathered,
        }
    }

    pub fn sharding(&self) -> OpSharding<1> {
        let (input, output) = match *self {
            Self::Local { sharding } => (sharding, sharding),
            Self::Split { from_dim, to_dim } => (
                Sharding::Split { dim: from_dim },
                Sharding::Split { dim: to_dim },
            ),
            Self::Gathered => (Sharding::Replicated, Sharding::Replicated),
        };
        OpSharding {
            inputs: [input],
            output,
        }
    }

    /// The output dim starting where input dim `dim` starts in memory. Its chunks hold the same
    /// elements when it has the same length, or when both lengths split evenly over the members
    /// and leave none empty.
    fn chunk_preserving_dim(
        from: &Shape,
        to: &Shape,
        dim: usize,
        num_members: usize,
    ) -> Option<usize> {
        let before = from[..dim].iter().product::<usize>();
        let len = from[dim];
        let mut prefix = 1;
        for (candidate, &size) in to.iter().enumerate() {
            if prefix == before && size != 1 {
                let same_chunks = size == len
                    || (len.is_multiple_of(num_members)
                        && size.is_multiple_of(num_members)
                        && size >= num_members);
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
                from_dim: 2,
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
                from_dim: 2,
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
                from_dim: 0,
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
                from_dim: 2,
                to_dim: 1
            }
        );
    }
}
