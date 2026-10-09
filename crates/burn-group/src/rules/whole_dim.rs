use crate::{GroupPlacement, OpPlacement, Reduction};

/// For an op that reads whole dims at once and is not linear, like softmax or max: it runs on
/// every member only when none of those dims is split.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WholeDimRule {
    /// Each member runs the op on its shard, and the output keeps the input's placement.
    Local { placement: GroupPlacement },
    /// A dim it reads is split, or the input is partial: the input is gathered first.
    Gathered,
}

impl WholeDimRule {
    pub fn new(input: GroupPlacement, reads: Reduction) -> Self {
        match input {
            GroupPlacement::Partial => Self::Gathered,
            GroupPlacement::Sharded { dim } if reads.contains(dim) => Self::Gathered,
            placement => Self::Local { placement },
        }
    }

    pub fn placement(&self) -> OpPlacement<1> {
        let placement = match *self {
            Self::Local { placement } => placement,
            Self::Gathered => GroupPlacement::Replicated,
        };
        OpPlacement {
            inputs: [placement],
            output: placement,
        }
    }
}
