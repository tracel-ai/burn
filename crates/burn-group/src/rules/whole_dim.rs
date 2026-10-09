use crate::{OpPlacement, Placement, Reduction};

/// For an op that reads whole dims at once and is not linear, like softmax or max: it runs on
/// every rank only when none of those dims is split.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WholeDimRule {
    /// Each rank runs the op on its shard, and the output keeps the input's placement.
    Local { placement: Placement },
    /// A dim it reads is split, or the input is partial: the input is gathered first.
    Gathered,
}

impl WholeDimRule {
    pub fn new(input: Placement, reads: Reduction) -> Self {
        match input {
            Placement::Partial => Self::Gathered,
            Placement::Sharded { dim } if reads.contains(dim) => Self::Gathered,
            placement => Self::Local { placement },
        }
    }

    pub fn placement(&self) -> OpPlacement<1> {
        let placement = match *self {
            Self::Local { placement } => placement,
            Self::Gathered => Placement::Replicated,
        };
        OpPlacement {
            inputs: [placement],
            output: placement,
        }
    }
}
