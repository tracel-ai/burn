use crate::{OpSharding, Sharding};

/// The dims a reduce op collapses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReducedDims {
    /// Every dim, down to a single element.
    All,
    /// One dim, kept with length 1.
    Dim(usize),
}

impl ReducedDims {
    pub fn contains(self, dim: usize) -> bool {
        match self {
            ReducedDims::All => true,
            ReducedDims::Dim(reduced) => reduced == dim,
        }
    }
}

/// A sum is linear: a partial input stays partial, and reducing the split dim leaves one summand
/// per member.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SumRule {
    /// Each member reduces its shard, and the output keeps the input's sharding.
    Local { sharding: Sharding },
    /// The split dim is reduced: each member's result is a summand.
    AcrossMembers { dim: usize },
}

impl SumRule {
    pub fn new(input: Sharding, reduced_dims: ReducedDims) -> Self {
        match input {
            Sharding::Split { dim } if reduced_dims.contains(dim) => Self::AcrossMembers { dim },
            sharding => Self::Local { sharding },
        }
    }

    pub fn sharding(&self) -> OpSharding<1> {
        let (input, output) = match *self {
            Self::Local { sharding } => (sharding, sharding),
            Self::AcrossMembers { dim } => (Sharding::Split { dim }, Sharding::PartialSum),
        };
        OpSharding {
            inputs: [input],
            output,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Sharding::{PartialSum, Split};

    #[test]
    fn reducing_the_split_dim_leaves_a_summand_per_member() {
        let rule = SumRule::new(Split { dim: 1 }, ReducedDims::Dim(1));

        assert_eq!(rule.sharding().output, PartialSum);
    }

    #[test]
    fn reducing_another_dim_keeps_the_split() {
        let rule = SumRule::new(Split { dim: 1 }, ReducedDims::Dim(0));

        assert_eq!(rule.sharding().output, Split { dim: 1 });
    }
}
