use crate::{
    GroupPlacement::{self, Partial, Replicated, Sharded},
    OpPlacement,
};

/// Which part of an embedding lookup is split. The weights are `[vocab, hidden]` and the
/// output `[batch, seq, hidden]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddingRule {
    /// Nothing is split: every rank looks up every row.
    Unsplit,
    /// The weights are split by hidden column, and so is the output.
    Columns,
    /// The weights are split by vocab row: each rank looks up the rows it holds and zeros the
    /// rest, so the output is a partial sum.
    Vocab,
    /// The indices are split along `dim`, and so is the output.
    Indices { dim: usize },
    /// Partial weights look up partial rows.
    PartialWeights,
    /// No split fits: both are gathered.
    Gathered,
}

impl EmbeddingRule {
    pub const VOCAB_DIM: usize = 0;
    pub const HIDDEN_DIM: usize = 1;
    pub const OUTPUT_HIDDEN_DIM: usize = 2;
    pub const VOCAB_ROWS: GroupPlacement = Sharded {
        dim: Self::VOCAB_DIM,
    };
    pub const HIDDEN_COLUMNS: GroupPlacement = Sharded {
        dim: Self::HIDDEN_DIM,
    };
    pub const OUTPUT_COLUMNS: GroupPlacement = Sharded {
        dim: Self::OUTPUT_HIDDEN_DIM,
    };

    pub fn new(weights: GroupPlacement, indices: GroupPlacement) -> Self {
        match (weights, indices) {
            (Replicated, Replicated) => Self::Unsplit,
            (Replicated, Sharded { dim }) => Self::Indices { dim },
            (Self::HIDDEN_COLUMNS, _) => Self::Columns,
            (Self::VOCAB_ROWS, _) => Self::Vocab,
            (Partial, _) => Self::PartialWeights,
            _ => Self::Gathered,
        }
    }

    /// Inputs: weights, indices.
    pub fn placement(&self) -> OpPlacement<2> {
        let (inputs, output) = match *self {
            Self::Unsplit | Self::Gathered => ([Replicated, Replicated], Replicated),
            Self::Columns => ([Self::HIDDEN_COLUMNS, Replicated], Self::OUTPUT_COLUMNS),
            Self::Vocab => ([Self::VOCAB_ROWS, Replicated], Partial),
            Self::Indices { dim } => ([Replicated, Sharded { dim }], Sharded { dim }),
            Self::PartialWeights => ([Partial, Replicated], Partial),
        };
        OpPlacement { inputs, output }
    }
}

/// The gradient of the embedding weights, from the output's gradient and the indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddingBackwardRule {
    /// Nothing is split: every rank computes the whole gradient.
    Unsplit,
    /// The weights are split by vocab row: each rank sums the gradient of the tokens in its
    /// chunk, so the gradient is split like the weights.
    Vocab,
    /// The output gradient is split by hidden column, and so is the weights' gradient.
    Columns,
    /// The tokens are split along `dim`: each rank sums the gradient of its own tokens.
    Indices { dim: usize },
    /// A partial output gradient gives a partial weights gradient.
    PartialGrad,
}

impl EmbeddingBackwardRule {
    /// A vocab split is kept even against a split output gradient: gathering the gradient of
    /// the tokens costs less than gathering the table.
    pub fn new(
        weights: GroupPlacement,
        output_grad: GroupPlacement,
        indices: GroupPlacement,
    ) -> Self {
        match (weights, output_grad, indices) {
            (EmbeddingRule::VOCAB_ROWS, _, _) => Self::Vocab,
            (_, EmbeddingRule::OUTPUT_COLUMNS, _) => Self::Columns,
            (_, Sharded { dim }, _) | (_, Replicated, Sharded { dim }) => Self::Indices { dim },
            (_, Partial, _) => Self::PartialGrad,
            (_, Replicated, _) => Self::Unsplit,
        }
    }

    /// Inputs: weights (read for their shape only), output gradient, indices.
    pub fn placement(&self) -> OpPlacement<3> {
        let (inputs, output) = match *self {
            Self::Unsplit => ([Replicated, Replicated, Replicated], Replicated),
            Self::Vocab => (
                [EmbeddingRule::VOCAB_ROWS, Replicated, Replicated],
                EmbeddingRule::VOCAB_ROWS,
            ),
            Self::Columns => (
                [
                    EmbeddingRule::HIDDEN_COLUMNS,
                    EmbeddingRule::OUTPUT_COLUMNS,
                    Replicated,
                ],
                EmbeddingRule::HIDDEN_COLUMNS,
            ),
            Self::Indices { dim } => ([Replicated, Sharded { dim }, Sharded { dim }], Partial),
            Self::PartialGrad => ([Replicated, Partial, Replicated], Partial),
        };
        OpPlacement { inputs, output }
    }
}
