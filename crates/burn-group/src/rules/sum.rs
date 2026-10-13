use crate::{OpSharding, Sharding};

/// The dims a sum reduces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SummedDims {
    /// Every dim, down to a single element.
    All,
    /// One dim, kept with length 1.
    Dim(usize),
}

impl SummedDims {
    pub fn contains(self, dim: usize) -> bool {
        match self {
            SummedDims::All => true,
            SummedDims::Dim(summed) => summed == dim,
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
    pub fn new(input: Sharding, summed_dims: SummedDims) -> Self {
        match input {
            Sharding::Split { dim } if summed_dims.contains(dim) => Self::AcrossMembers { dim },
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
        let rule = SumRule::new(Split { dim: 1 }, SummedDims::Dim(1));

        assert_eq!(rule.sharding().output, PartialSum);
    }

    #[test]
    fn reducing_another_dim_keeps_the_split() {
        let rule = SumRule::new(Split { dim: 1 }, SummedDims::Dim(0));

        assert_eq!(rule.sharding().output, Split { dim: 1 });
    }
}
