//! Deciding which axis a tensor's vectorized accesses run along.
//!
//! A vector of `width` elements is only a vector if those elements are adjacent in
//! memory. Which axis that holds along is a property of the layout the block
//! iterates in, not of the tensor's rank — and once a block is free to choose a
//! permuted layout, the two stop coinciding.
//!
//! Every tensor a block touches has to be measured along the *same* axis the block
//! iterates along, or a vector of the reference and a vector of that tensor cover
//! different elements. So each block casts a vote per tensor here, the votes are
//! merged, and anything that cannot be lined up is refused outright.

use super::super::{ReferenceSelection, layout::permuted_innermost_dim};

/// What one block asks of one tensor's vectorization axis.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AxisVote {
    /// The block iterates in logical dimension order, so the tensor keeps the
    /// default axis — its own last dimension.
    Default,
    /// The block iterates in a permuted order innermost along this dimension, and
    /// every tensor it touches has to be measured along the same one.
    Along(usize),
    /// The tensor cannot be lined up with the order the block iterates in, and must
    /// not be vectorized.
    Never,
}

impl AxisVote {
    /// The vote that holds for a tensor two blocks both touch.
    ///
    /// Anything but the same answer twice leaves the tensor unvectorized: the
    /// disagreement means at least one of the two blocks would be vectorizing
    /// against a layout that is not its own.
    pub fn merged_with(self, other: Self) -> Self {
        match self == other {
            true => self,
            false => AxisVote::Never,
        }
    }
}

/// The dimension a permuted reference advances a line along, or `None` when the
/// block iterates in logical dimension order.
///
/// A virtual reference indexes through a transform and those paths were written
/// against the last dimension; a contiguous concrete one advances along the last
/// dimension by construction. Neither asks anything of the tensors it touches
/// beyond the default, so both answer `None`.
pub fn permuted_innermost(reference: &ReferenceSelection) -> Option<usize> {
    let ReferenceSelection::Concrete { shape, strides, .. } = reference else {
        return None;
    };

    permuted_innermost_dim(shape, strides)
}

/// What a block whose reference is innermost along `permuted` asks of one tensor.
///
/// `strides` is given only for tensors the vectorization pass will not check them
/// for itself, which is what makes the difference between refusing and trusting it
/// to refuse.
pub fn axis_vote(permuted: Option<usize>, rank: usize, strides: Option<&[usize]>) -> AxisVote {
    let Some(axis) = permuted else {
        return AxisVote::Default;
    };

    // A reshaped view can have fewer dimensions than the reference. There is no
    // dimension of its own to line up with the reference's, and reading its shape
    // at the reference's innermost index would index past the end.
    if axis >= rank {
        return AxisVote::Never;
    }

    // A line along a dimension the tensor is not contiguous in covers elements that
    // are not adjacent in its buffer, so it is not a line at all.
    if let Some(strides) = strides
        && strides[axis] != 1
    {
        return AxisVote::Never;
    }

    AxisVote::Along(axis)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::codegen::ir::{FuseArg, FuseType, LayoutInfo};
    use burn_std::{Shape, Strides};

    fn concrete(shape: &[usize], strides: &[usize]) -> ReferenceSelection {
        ReferenceSelection::Concrete {
            layout: FuseArg::Output(0, FuseType::F32, LayoutInfo::IsRef),
            shape: Shape::from(shape.to_vec()),
            strides: Strides::new(strides),
        }
    }

    #[test]
    fn a_contiguous_reference_asks_for_nothing() {
        // The whole point: a block that iterates in logical dimension order has to
        // keep behaving exactly as it did before a block could choose its layout.
        let reference = concrete(&[2, 48, 16, 16], &[48 * 16 * 16, 16 * 16, 16, 1]);

        assert_eq!(permuted_innermost(&reference), None);
        assert_eq!(axis_vote(None, 4, None), AxisVote::Default);
    }

    #[test]
    fn a_virtual_reference_asks_for_nothing() {
        let reference = ReferenceSelection::Reshaped { reshape_pos: 0 };
        assert_eq!(permuted_innermost(&reference), None);

        let reference = ReferenceSelection::Searching;
        assert_eq!(permuted_innermost(&reference), None);
    }

    #[test]
    fn a_reference_that_is_not_dense_asks_for_nothing() {
        // A padded or sliced reference has no dimension order to impose.
        assert_eq!(permuted_innermost(&concrete(&[4, 8], &[16, 1])), None);
    }

    #[test]
    fn an_nhwc_reference_asks_for_its_channel_dimension() {
        let reference = concrete(&[2, 48, 16, 16], &[16 * 16 * 48, 1, 16 * 48, 48]);

        assert_eq!(permuted_innermost(&reference), Some(1));
        assert_eq!(axis_vote(Some(1), 4, None), AxisVote::Along(1));
    }

    #[test]
    fn a_permuted_reference_skips_its_degenerate_dimensions() {
        // The dimension order ends at a size-one dimension whose stride is zero.
        // Advancing a line along it would read the same element `width` times.
        let reference = concrete(&[2, 48, 16, 1], &[768, 1, 48, 0]);

        assert_eq!(permuted_innermost(&reference), Some(1));
    }

    #[test]
    fn an_output_the_block_cannot_write_in_lines_is_refused() {
        // A reshaped output on the recompute path is allocated contiguous while its
        // block iterates NHWC. `vectorization_output` is handed a shape and never
        // sees a stride, so nothing downstream would catch this.
        let contiguous = [48 * 16 * 16, 16 * 16, 16, 1];

        assert_eq!(
            axis_vote(Some(1), 4, Some(&contiguous)),
            AxisVote::Never,
            "stride along the block's innermost dimension is not one",
        );

        let nhwc = [16 * 16 * 48, 1, 16 * 48, 48];
        assert_eq!(axis_vote(Some(1), 4, Some(&nhwc)), AxisVote::Along(1));
    }

    #[test]
    fn an_output_of_a_lower_rank_is_refused_rather_than_indexed_past_its_end() {
        // A reshape can drop dimensions. Reading `shape[axis]` at the reference's
        // innermost index would panic; the default axis could never do that, since
        // it came from the tensor's own rank.
        assert_eq!(axis_vote(Some(3), 2, None), AxisVote::Never);
        assert_eq!(axis_vote(Some(2), 2, None), AxisVote::Never);
        assert_eq!(axis_vote(Some(1), 2, None), AxisVote::Along(1));
    }

    #[test]
    fn an_input_is_trusted_to_refuse_an_axis_it_cannot_use() {
        // Inputs are voted without strides on purpose: `vectorization_input` makes
        // the same check, and unlike this one it tells a broadcast dimension from a
        // dimension it merely cannot line up.
        assert_eq!(axis_vote(Some(1), 4, None), AxisVote::Along(1));
    }

    #[test]
    fn blocks_that_disagree_cancel_out() {
        // Any two votes that are not the same answer leave the tensor unvectorized,
        // including a permuted block meeting one that only wants the default.
        assert_eq!(
            AxisVote::Along(1).merged_with(AxisVote::Along(3)),
            AxisVote::Never
        );
        assert_eq!(
            AxisVote::Along(1).merged_with(AxisVote::Default),
            AxisVote::Never
        );
        assert_eq!(
            AxisVote::Default.merged_with(AxisVote::Default),
            AxisVote::Default
        );
        assert_eq!(
            AxisVote::Along(1).merged_with(AxisVote::Along(1)),
            AxisVote::Along(1)
        );
    }

    #[test]
    fn a_refusal_survives_every_later_vote() {
        // Votes are merged in whatever order the blocks happen to come in, so a
        // refusal has to be absorbing rather than merely first.
        for vote in [AxisVote::Default, AxisVote::Along(1), AxisVote::Never] {
            assert_eq!(AxisVote::Never.merged_with(vote), AxisVote::Never);
            assert_eq!(vote.merged_with(AxisVote::Never), AxisVote::Never);
        }
    }
}
