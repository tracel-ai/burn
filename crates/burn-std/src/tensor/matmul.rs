use crate::tensor::{Metadata, Shape, Strides};

/// Accelerated matmul tiles consume rows in chunks of this size; a row count
/// that is a multiple of it already tiles cleanly.
const ROW_TILE: usize = 32;

/// The action to take for a batched matmul with a broadcast rhs.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum MatmulTransformAction {
    /// Launch the matmul as-is.
    Keep,
    /// Fold every lhs/out batch dim into the rows: `[.., b, m, k] @ [.., 1, k, n]`
    /// runs as one `[b*m, k] @ [k, n]` matmul against the shared rhs, instead of
    /// `b` matmuls that each re-read it.
    MergeBatches {
        /// The merged row count (batches × rows).
        rows: usize,
    },
}

impl MatmulTransformAction {
    /// Apply the action to one merged operand (lhs or out): fold the batch dims
    /// into the rows, in place. A pure metadata rewrite — the buffer is shared.
    ///
    /// Requires batch-contiguous rows (see [MatmulTransformAnalysis]); the merge
    /// keeps the rank, with every batch dim set to 1.
    pub fn apply(&self, meta: &mut Metadata) {
        let rows = match self {
            MatmulTransformAction::Keep => return,
            MatmulTransformAction::MergeBatches { rows } => *rows,
        };

        let rank = meta.rank();
        let stride_rows = merged_row_stride(meta.shape(), meta.strides())
            .expect("The action requires batch-contiguous rows");

        for i in 0..rank - 2 {
            meta.shape_mut()[i] = 1;
            meta.strides_mut()[i] = rows * stride_rows;
        }
        meta.shape_mut()[rank - 2] = rows;
        meta.strides_mut()[rank - 2] = stride_rows;
    }
}

/// The facts of a batched matmul `lhs @ rhs` relevant to folding its batches
/// into its rows.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct MatmulTransformAnalysis {
    /// Product of the lhs batch dims.
    batches: usize,
    /// Rows of one batch (`m`).
    rows: usize,
    /// Columns of the output (`n`).
    cols: usize,
    /// The rhs is shared by every batch, and each operand's rows advance with a
    /// single stride across batch boundaries (no holes), so the fold is a pure
    /// view.
    mergeable: bool,
}

impl MatmulTransformAnalysis {
    /// Analyze from shapes alone: the operands are assumed batch-contiguous, as
    /// holds for freshly materialized tensors. Use [Self::from_metadata] when
    /// the layouts are known.
    ///
    /// The rhs may have a lower rank than the lhs (implicit broadcast).
    pub fn from_shapes(lhs: &Shape, rhs: &Shape) -> Self {
        Self::new(lhs, rhs, true)
    }

    /// Analyze from full metadata: on top of the shape facts, the lhs and out
    /// rows must advance with a single stride across batch boundaries.
    pub fn from_metadata(lhs: &Metadata, rhs: &Metadata, out: &Metadata) -> Self {
        let mergeable = merged_row_stride(lhs.shape(), lhs.strides()).is_some()
            && merged_row_stride(out.shape(), out.strides()).is_some();

        Self::new(lhs.shape(), rhs.shape(), mergeable)
    }

    fn new(lhs: &Shape, rhs: &Shape, rows_contiguous: bool) -> Self {
        let rank = lhs.num_dims();
        let rank_rhs = rhs.num_dims();

        let batches = lhs[..rank - 2].iter().product();
        let rows = lhs[rank - 2];
        let cols = rhs[rank_rhs - 1];

        // Every batch dim the rhs actually has must be broadcast.
        let rhs_shared = rhs[..rank_rhs - 2].iter().all(|&dim| dim == 1);

        Self {
            batches,
            rows,
            cols,
            mergeable: rhs_shared && rows_contiguous,
        }
    }
}

/// Decides the [action](MatmulTransformAction) to take for a batched matmul,
/// given its [analysis](MatmulTransformAnalysis).
#[derive(Debug, Default, Clone, Copy)]
pub enum MatmulTransformPolicy {
    /// Merge the batches into the rows when the fold is a pure view and the
    /// merged problem tiles better: per-batch rows that already fill row tiles
    /// stay batched (a batched matmul beats one big matmul at that scale), tiny
    /// row counts merge when that brings the output closer to square.
    #[default]
    BetterTiling,
    /// Never transform.
    Never,
}

impl MatmulTransformPolicy {
    /// The action to take for a matmul with the given analysis.
    pub fn action(&self, analysis: &MatmulTransformAnalysis) -> MatmulTransformAction {
        match self {
            MatmulTransformPolicy::Never => MatmulTransformAction::Keep,
            MatmulTransformPolicy::BetterTiling => {
                if !analysis.mergeable || analysis.batches == 1 {
                    return MatmulTransformAction::Keep;
                }

                // Rows already fill the tiles: keep the batched form.
                if analysis.rows.is_multiple_of(ROW_TILE) {
                    return MatmulTransformAction::Keep;
                }

                let rows = analysis.batches * analysis.rows;

                // The merge must bring the output closer to square.
                if !squarer(rows, analysis.rows, analysis.cols) {
                    return MatmulTransformAction::Keep;
                }

                MatmulTransformAction::MergeBatches { rows }
            }
        }
    }
}

/// Whether an output of `rows_new` x `cols` is closer to square than
/// `rows` x `cols`.
fn squarer(rows_new: usize, rows: usize, cols: usize) -> bool {
    // min(a, c) / max(a, c) > min(b, c) / max(b, c), cross-multiplied to stay
    // in integers.
    rows_new.min(cols) * rows.max(cols) > rows.min(cols) * rows_new.max(cols)
}

/// The single stride with which rows advance across batch boundaries, when the
/// batch and row dims form a hole-free chain — the requirement for folding the
/// batches into the rows as a pure view. Dims of size 1 carry arbitrary strides
/// and never anchor the chain.
fn merged_row_stride(shape: &Shape, strides: &Strides) -> Option<usize> {
    let rank = shape.num_dims();
    let mut chained: Option<(usize, usize)> = None;

    for i in (0..rank - 1).rev() {
        if shape[i] == 1 {
            continue;
        }
        match chained {
            None => chained = Some((strides[i], strides[i] * shape[i])),
            Some((row_stride, expected)) => {
                if strides[i] != expected {
                    return None;
                }
                chained = Some((row_stride, strides[i] * shape[i]));
            }
        }
    }

    match chained {
        Some((row_stride, _)) => Some(row_stride),
        // Every merged dim is a unit dim: a single row, any stride works.
        None => Some(shape[rank - 1] * strides[rank - 1]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape;

    fn analysis(lhs: &[usize], rhs: &[usize]) -> MatmulTransformAnalysis {
        MatmulTransformAnalysis::from_shapes(&Shape::from(lhs), &Shape::from(rhs))
    }

    fn action(analysis: &MatmulTransformAnalysis) -> MatmulTransformAction {
        MatmulTransformPolicy::default().action(analysis)
    }

    #[test]
    fn merges_batched_vec_mat() {
        // A decode step: 16 broadcast vec-mats fold into one 16-row matmul.
        let analysis = analysis(&[16, 1, 4096], &[4096, 14336]);

        assert_eq!(
            action(&analysis),
            MatmulTransformAction::MergeBatches { rows: 16 }
        );
    }

    #[test]
    fn keeps_row_tiled_batches() {
        // A prefill step: 512 rows per batch already fill the row tiles; the
        // batched matmul beats one big matmul.
        let analysis = analysis(&[8, 512, 4096], &[4096, 14336]);

        assert_eq!(action(&analysis), MatmulTransformAction::Keep);
    }

    #[test]
    fn keeps_single_batch() {
        let analysis = analysis(&[1, 1, 4096], &[4096, 14336]);

        assert_eq!(action(&analysis), MatmulTransformAction::Keep);
    }

    #[test]
    fn keeps_matrix() {
        let analysis = analysis(&[16, 4096], &[4096, 14336]);

        assert_eq!(action(&analysis), MatmulTransformAction::Keep);
    }

    #[test]
    fn keeps_batched_rhs() {
        // The rhs differs per batch: nothing is shared, nothing to fold.
        let analysis = analysis(&[8, 1, 64], &[8, 64, 32]);

        assert_eq!(action(&analysis), MatmulTransformAction::Keep);
    }

    #[test]
    fn merges_with_broadcast_rhs_rank() {
        let analysis = analysis(&[8, 1, 64], &[1, 64, 32]);

        assert_eq!(
            action(&analysis),
            MatmulTransformAction::MergeBatches { rows: 8 }
        );
    }

    #[test]
    fn keeps_when_merge_leaves_square() {
        // 1000 rows tower over 64 cols; folding to 4000 only makes the output
        // less square.
        let analysis = analysis(&[4, 1000, 64], &[64, 64]);

        assert_eq!(action(&analysis), MatmulTransformAction::Keep);
    }

    #[test]
    fn merges_multiple_batch_dims() {
        let analysis = analysis(&[2, 8, 1, 4096], &[4096, 14336]);

        assert_eq!(
            action(&analysis),
            MatmulTransformAction::MergeBatches { rows: 16 }
        );
    }

    #[test]
    fn never_policy_keeps() {
        let analysis = analysis(&[16, 1, 4096], &[4096, 14336]);

        assert_eq!(
            MatmulTransformPolicy::Never.action(&analysis),
            MatmulTransformAction::Keep
        );
    }

    #[test]
    fn metadata_merges_pitched_rows() {
        // Padded batches with one row each still fold: the merged matrix reads
        // its rows with the single (pitched) stride 8192.
        let lhs = Metadata::new(shape![16, 1, 4096], crate::strides![8192, 4096, 1]);
        let rhs = Metadata::new(shape![1, 4096, 14336], crate::strides![0, 14336, 1]);
        let out = Metadata::new(shape![16, 1, 14336], crate::strides![14336, 14336, 1]);

        let analysis = MatmulTransformAnalysis::from_metadata(&lhs, &rhs, &out);

        assert_eq!(
            action(&analysis),
            MatmulTransformAction::MergeBatches { rows: 16 }
        );
    }

    #[test]
    fn metadata_keeps_batch_holes() {
        // Rows are contiguous within a batch but batches leave a gap: no single
        // row stride can express the fold.
        let lhs = Metadata::new(shape![4, 3, 8], crate::strides![48, 8, 1]);
        let rhs = Metadata::new(shape![1, 8, 32], crate::strides![0, 32, 1]);
        let out = Metadata::new(shape![4, 3, 32], crate::strides![96, 32, 1]);

        let analysis = MatmulTransformAnalysis::from_metadata(&lhs, &rhs, &out);

        assert_eq!(action(&analysis), MatmulTransformAction::Keep);
    }

    #[test]
    fn metadata_merges_contiguous() {
        let lhs = Metadata::new(shape![16, 1, 4096], crate::strides![4096, 4096, 1]);
        let rhs = Metadata::new(shape![1, 4096, 14336], crate::strides![0, 14336, 1]);
        let out = Metadata::new(shape![16, 1, 14336], crate::strides![14336, 14336, 1]);

        let analysis = MatmulTransformAnalysis::from_metadata(&lhs, &rhs, &out);

        assert_eq!(
            action(&analysis),
            MatmulTransformAction::MergeBatches { rows: 16 }
        );
    }

    #[test]
    fn metadata_ignores_unit_dim_strides() {
        // Size-1 dims carry arbitrary strides; they must not anchor the chain.
        let lhs = Metadata::new(shape![16, 1, 4096], crate::strides![4096, 12345, 1]);
        let rhs = Metadata::new(shape![1, 4096, 14336], crate::strides![0, 14336, 1]);
        let out = Metadata::new(shape![16, 1, 14336], crate::strides![14336, 99999, 1]);

        let analysis = MatmulTransformAnalysis::from_metadata(&lhs, &rhs, &out);

        assert_eq!(
            action(&analysis),
            MatmulTransformAction::MergeBatches { rows: 16 }
        );
    }

    #[test]
    fn apply_folds_batches_in_place() {
        let mut lhs = Metadata::new(shape![16, 1, 4096], crate::strides![4096, 4096, 1]);

        MatmulTransformAction::MergeBatches { rows: 16 }.apply(&mut lhs);

        assert_eq!(lhs.shape(), &shape![1, 16, 4096]);
        assert_eq!(lhs.strides()[1], 4096);
        assert_eq!(lhs.strides()[2], 1);
    }

    #[test]
    fn apply_keep_is_noop() {
        let mut lhs = Metadata::new(shape![16, 1, 4096], crate::strides![4096, 4096, 1]);

        MatmulTransformAction::Keep.apply(&mut lhs);

        assert_eq!(lhs.shape(), &shape![16, 1, 4096]);
    }
}
