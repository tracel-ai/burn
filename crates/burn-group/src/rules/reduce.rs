use crate::{OpPlacement, Placement};

/// The dims an op reduces, or reads whole.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reduction {
    /// Every dim, down to a single element.
    All,
    /// One dim, kept with length 1.
    Dim(usize),
}

impl Reduction {
    pub fn contains(self, dim: usize) -> bool {
        match self {
            Reduction::All => true,
            Reduction::Dim(reduced) => reduced == dim,
        }
    }
}

/// For a float sum or mean: linear, so a partial input stays partial, and a reduction over the
/// split dim leaves one summand per rank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceRule {
    /// Each rank reduces its shard, and the output keeps the input's placement.
    Local { placement: Placement },
    /// The split dim is reduced: each rank's result is a summand.
    AcrossRanks { dim: usize },
}

impl ReduceRule {
    pub fn new(input: Placement, reduction: Reduction) -> Self {
        match input {
            Placement::Sharded { dim } if reduction.contains(dim) => Self::AcrossRanks { dim },
            placement => Self::Local { placement },
        }
    }

    pub fn placement(&self) -> OpPlacement<1> {
        let (input, output) = match *self {
            Self::Local { placement } => (placement, placement),
            Self::AcrossRanks { dim } => (Placement::Sharded { dim }, Placement::Partial),
        };
        OpPlacement {
            inputs: [input],
            output,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Placement::{Partial, Sharded};

    #[test]
    fn reducing_the_split_dim_leaves_a_summand_per_rank() {
        let rule = ReduceRule::new(Sharded { dim: 1 }, Reduction::Dim(1));

        assert_eq!(rule.placement().output, Partial);
    }

    #[test]
    fn reducing_another_dim_keeps_the_split() {
        let rule = ReduceRule::new(Sharded { dim: 1 }, Reduction::Dim(0));

        assert_eq!(rule.placement().output, Sharded { dim: 1 });
    }
}
