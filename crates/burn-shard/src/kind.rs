use burn_std::Slice;

/// What an op computes, as far as its sharding can tell: the planner derives an op's rule from
/// its kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpKind<'a> {
    /// Per element: each input has the output's dims, of length 1 where it broadcasts.
    Elementwise(Linearity),
    /// Reads `dim` of its one input whole and treats every other dim independently, like
    /// softmax or cumsum.
    AlongDim { dim: usize },
    /// Collapses `dims` of its one input.
    Reduction {
        dims: ReducedDims<'a>,
        kind: ReductionKind,
    },
    /// The matmul of its two inputs.
    Matmul,
    /// Drawn from the RNG, with no inputs.
    Random,
    /// Its input read in the output's shape.
    Reshape,
    /// Its input with two dims swapped.
    SwapDims { dim1: usize, dim2: usize },
    /// Its input with its dims in the order of `axes`.
    Permute { axes: &'a [usize] },
    /// The part of its input within `ranges`.
    Slice { ranges: &'a [Slice] },
    /// Its first input with its second written within `ranges`.
    SliceAssign { ranges: &'a [Slice] },
    /// Its input broadcast along its dims of length 1, and along new leading dims.
    Expand,
    /// Attention of a query, key and value, then an optional mask or bias, or both.
    Attention,
}

/// Which inputs of an elementwise op a partial sum can pass through: its output stays partial
/// only where the op distributes over the sum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Linearity {
    /// Linear in no input, like relu: a partial input is summed first.
    Nonlinear,
    /// Linear in every input at once: the output stays partial when every input is, like add.
    Linear,
    /// Linear in each input with the others fixed: one partial input among replicated ones
    /// keeps the output partial, like mul.
    Multilinear,
    /// Linear in one input with the others replicated, like a dividend.
    LinearIn { input: usize },
}

/// The dims a reduce op collapses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReducedDims<'a> {
    /// Every dim, down to a single element.
    All,
    /// These dims, each kept with length 1.
    Dims(&'a [usize]),
}

impl ReducedDims<'_> {
    pub fn contains(self, dim: usize) -> bool {
        match self {
            ReducedDims::All => true,
            ReducedDims::Dims(reduced) => reduced.contains(&dim),
        }
    }
}

/// What a reduce op computes, as far as a sharding can tell them apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReductionKind {
    /// The sum of a split dim is the sum of each member's sum.
    Sum,
    /// Linear like a sum, but each member's mean over its own shard is not a summand of the
    /// whole mean.
    Mean,
    /// Max, min, prod, argmax and the like: neither passes a partial sum through nor splits into
    /// summands.
    Nonlinear,
}

impl ReductionKind {
    pub fn is_linear(self) -> bool {
        matches!(self, ReductionKind::Sum | ReductionKind::Mean)
    }
}
