use smallvec::smallvec;

use crate::{OpSharding, ReducedDims, ReductionKind, Sharding};

/// How the members run a reduce op.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceRule {
    /// Each member reduces its shard, and the output keeps the input's sharding.
    Local { sharding: Sharding },
    /// A sum reduces the split dim: each member's result is a summand.
    AcrossMembers { dim: usize },
    /// The input is gathered first: this kind of reduction cannot combine the members' results.
    Gathered,
}

impl ReduceRule {
    pub fn new(input: Sharding, reduced_dims: ReducedDims, reduction: ReductionKind) -> Self {
        match input {
            Sharding::Split { dim } if reduced_dims.contains(dim) => match reduction {
                ReductionKind::Sum => Self::AcrossMembers { dim },
                ReductionKind::Mean | ReductionKind::Nonlinear => Self::Gathered,
            },
            // An all-reduce over every member would count each group's chunk once per member.
            Sharding::ReplicatedSplit { dim, .. } if reduced_dims.contains(dim) => Self::Gathered,
            Sharding::PartialSum if !reduction.is_linear() => Self::Gathered,
            sharding => Self::Local { sharding },
        }
    }

    pub fn sharding(&self) -> OpSharding {
        let (input, output) = match *self {
            Self::Local { sharding } => (sharding, sharding),
            Self::AcrossMembers { dim } => (Sharding::Split { dim }, Sharding::PartialSum),
            Self::Gathered => (Sharding::Replicated, Sharding::Replicated),
        };
        OpSharding {
            inputs: smallvec![input],
            output,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Sharding::{PartialSum, Replicated, ReplicatedSplit, Split};

    #[test]
    fn summing_the_split_dim_leaves_a_summand_per_member() {
        let rule = ReduceRule::new(
            Split { dim: 1 },
            ReducedDims::Dims(&[1]),
            ReductionKind::Sum,
        );

        assert_eq!(rule.sharding().output, PartialSum);
    }

    #[test]
    fn reducing_another_dim_keeps_the_split() {
        let rule = ReduceRule::new(
            Split { dim: 1 },
            ReducedDims::Dims(&[0]),
            ReductionKind::Nonlinear,
        );

        assert_eq!(rule.sharding().output, Split { dim: 1 });
    }

    #[test]
    fn a_mean_over_the_split_dim_gathers() {
        let rule = ReduceRule::new(Split { dim: 0 }, ReducedDims::All, ReductionKind::Mean);

        assert_eq!(rule, ReduceRule::Gathered);
    }

    #[test]
    fn a_partial_sum_passes_through_a_mean_but_not_a_max() {
        let dims = ReducedDims::Dims(&[0]);

        assert_eq!(
            ReduceRule::new(PartialSum, dims, ReductionKind::Mean)
                .sharding()
                .output,
            PartialSum
        );
        assert_eq!(
            ReduceRule::new(PartialSum, dims, ReductionKind::Nonlinear).sharding(),
            OpSharding {
                inputs: smallvec![Replicated],
                output: Replicated
            }
        );
    }

    #[test]
    fn summing_a_replicated_split_dim_gathers() {
        let kv = ReplicatedSplit { dim: 1, groups: 2 };

        assert_eq!(
            ReduceRule::new(kv, ReducedDims::Dims(&[1]), ReductionKind::Sum),
            ReduceRule::Gathered
        );
        assert_eq!(
            ReduceRule::new(kv, ReducedDims::Dims(&[0]), ReductionKind::Sum)
                .sharding()
                .output,
            kv
        );
    }
}
