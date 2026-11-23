//! Deciding which axis each tensor's vectorized accesses run along.
//!
//! A vector of `width` elements is only a vector if those elements are adjacent
//! in memory. Which axis that holds along is a property of the layout the block
//! iterates in, not of the tensor's rank — and once a block is free to choose a
//! permuted layout, the two stop coinciding.
//!
//! Every tensor a block touches has to be measured along the *same* axis the
//! block iterates along, or a vector of the reference and a vector of that tensor
//! cover different elements. So the facts are gathered per tensor
//! ([VectorAxisAnalysis]), a rule turns them into a decision
//! ([VectorAxisPolicy]), and the decision is applied ([VectorAxisAction]).

use super::super::{
    HandleOutput, LaunchPlan,
    runner::{Vectorization, VectorizationAxis},
};
use super::Vect;
use crate::{
    CubeFusionHandle,
    engine::trace::{FuseResources, TensorView},
};
use burn_fusion::stream::Context;
use burn_ir::TensorId;
use std::collections::{BTreeMap, HashMap};

/// What to do with one tensor's vectorization axis.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VectorAxisAction {
    /// Leave the tensor on the default axis — its own last dimension.
    Default,
    /// Measure and step the tensor along this axis instead.
    Along(usize),
    /// Pin the tensor to a vector size of one: nothing lines it up with the axis
    /// its block iterates along.
    Refuse,
}

impl VectorAxisAction {
    /// The action that holds for a tensor two blocks both touch.
    ///
    /// Anything but the same answer twice refuses the tensor: the disagreement
    /// means at least one of the two blocks would be vectorizing against a layout
    /// that is not its own. A refusal therefore absorbs whatever is merged into
    /// it, which matters because blocks are analysed in whatever order they come.
    pub fn merged_with(self, other: Self) -> Self {
        match self == other {
            true => self,
            false => VectorAxisAction::Refuse,
        }
    }
}

/// The facts about one tensor, as one block sees it, that decide its axis.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct VectorAxisAnalysis {
    /// The axis the block iterates along, when it iterates in a permuted order.
    block_axis: Option<usize>,
    /// The tensor's own rank.
    rank: usize,
    /// The tensor's stride along the block's axis, for the tensors whose strides
    /// nothing downstream will check.
    stride: Option<usize>,
}

impl VectorAxisAnalysis {
    /// The facts for a tensor the block reads.
    ///
    /// No stride: [vectorization_input](super::base) makes that check itself, and
    /// unlike this analysis it tells a broadcast dimension from one it merely
    /// cannot line up — a distinction worth keeping, since a broadcast operand
    /// does not constrain the block's width and a refused one does.
    pub fn read(block_axis: Option<usize>, rank: usize) -> Self {
        Self {
            block_axis,
            rank,
            stride: None,
        }
    }

    /// The facts for a tensor the block writes.
    ///
    /// [vectorization_output](super::base) is handed a shape and never sees a
    /// stride, so an output allocated in an order its block does not iterate in
    /// has to be caught here or not at all.
    pub fn written(block_axis: Option<usize>, rank: usize, strides: &[usize]) -> Self {
        Self {
            block_axis,
            rank,
            stride: block_axis.and_then(|axis| strides.get(axis).copied()),
        }
    }
}

/// Decides the [action](VectorAxisAction) for one tensor, given its
/// [analysis](VectorAxisAnalysis).
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub enum VectorAxisPolicy {
    /// Line every tensor up with the axis its block iterates along, and refuse the
    /// ones that cannot be.
    #[default]
    LineUpWithBlock,
    /// Every tensor keeps its own last dimension.
    ///
    /// What the pass did before a block could choose its layout, and still the
    /// answer for a plan where no block iterates in a permuted order — which is
    /// most of them.
    AlwaysDefault,
}

impl VectorAxisPolicy {
    /// The policy a plan needs.
    ///
    /// Unless some block iterates in a permuted order there is nothing to decide,
    /// and saying so up front skips the whole analysis.
    pub fn of_plan(plan: &LaunchPlan<'_>) -> Self {
        match plan.blocks.iter().any(|block| block.vector_axis.is_some()) {
            true => VectorAxisPolicy::LineUpWithBlock,
            false => VectorAxisPolicy::AlwaysDefault,
        }
    }

    /// The action to take for a tensor with the given analysis.
    pub fn action(&self, analysis: &VectorAxisAnalysis) -> VectorAxisAction {
        let axis = match self {
            VectorAxisPolicy::AlwaysDefault => return VectorAxisAction::Default,
            VectorAxisPolicy::LineUpWithBlock => match analysis.block_axis {
                Some(axis) => axis,
                None => return VectorAxisAction::Default,
            },
        };

        // A reshaped view can have fewer dimensions than the reference. There is
        // no axis of its own to line up with the block's, and reading its shape at
        // the block's axis would index past the end.
        if axis >= analysis.rank {
            return VectorAxisAction::Refuse;
        }

        // A vector along an axis the tensor is not contiguous in covers elements
        // that are not adjacent in its buffer, so it is not a vector at all.
        if let Some(stride) = analysis.stride
            && stride != 1
        {
            return VectorAxisAction::Refuse;
        }

        VectorAxisAction::Along(axis)
    }
}

/// The axis every tensor of a launch plan vectorizes along.
pub struct VectorAxes {
    axis: VectorizationAxis,
    refused: Refusals,
}

/// The tensors [VectorAxisAction::Refuse] applies to, pinned to a vector size of
/// one once the vectorization pass has run — it cannot see what refused them.
/// Usually empty.
#[derive(Default)]
pub struct Refusals {
    ids: Vec<TensorId>,
}

impl VectorAxes {
    /// Decide the axes for a plan.
    pub fn resolve<Runner: Vectorization>(
        runner: &Runner,
        resources: &FuseResources,
        context: &Context<CubeFusionHandle>,
        plan: &LaunchPlan<'_>,
    ) -> Self {
        // The runner knows better for its own operands — the matmul one places the
        // axis by matrix layout — so anything it sets wins, refusals included.
        let axis = runner.axis(plan);
        let policy = VectorAxisPolicy::of_plan(plan);

        if let VectorAxisPolicy::AlwaysDefault = policy {
            return Self {
                axis,
                refused: Refusals::default(),
            };
        }

        Self::from_actions(axis, Actions::gather(policy, resources, context, plan))
    }

    fn from_actions(mut axis: VectorizationAxis, actions: Actions) -> Self {
        const UNSET: usize = usize::MAX;
        let mut refused = Vec::new();

        for (id, (action, pinned)) in actions.per_tensor {
            if axis.get(id, || UNSET) != UNSET {
                continue;
            }

            match action {
                VectorAxisAction::Default => {}
                VectorAxisAction::Along(along) => axis.insert(id, along),
                VectorAxisAction::Refuse => refused.push(pinned),
            }
        }

        Self {
            axis,
            refused: Refusals { ids: refused },
        }
    }

    /// The axis map the vectorization pass measures against, and the refusals
    /// applied after it has run.
    pub fn split(self) -> (VectorizationAxis, Refusals) {
        (self.axis, self.refused)
    }
}

impl Refusals {
    /// Pin every refused tensor to a vector size of one.
    ///
    /// A tensor already judged broadcast is left alone: it is read element by
    /// element either way, and calling it aligned would drag the whole block's
    /// width down with it.
    pub fn apply(&self, vectorizations: &mut BTreeMap<TensorId, Vect>) {
        for id in self.ids.iter() {
            if !matches!(vectorizations.get(id), Some(Vect::Broadcasted)) {
                vectorizations.insert(*id, Vect::Aligned(1));
            }
        }
    }
}

/// The action each tensor collected so far, merged across the blocks that touch
/// it.
struct Actions {
    // TODO: allocated on every kernel launch. It cannot be reused by keeping it on
    // the planner — the planner, and the `FuseTraceLauncher` above it, are built
    // fresh per launch too; see the TODO on `FuseTraceLauncher` about reusing the
    // launcher and resetting its state. Only reached when a block actually
    // iterates in a permuted order, which is why it is not urgent.
    //
    // Keyed by the id the vectorization pass looks the axis up by, carrying the id
    // whose vector size a refusal has to be pinned on. The two differ for a view:
    // it is vectorized under its own id, but the verdict lands on the tensor it
    // views.
    per_tensor: HashMap<TensorId, (VectorAxisAction, TensorId)>,
}

impl Actions {
    fn gather(
        policy: VectorAxisPolicy,
        resources: &FuseResources,
        context: &Context<CubeFusionHandle>,
        plan: &LaunchPlan<'_>,
    ) -> Self {
        let mut actions = Self {
            per_tensor: HashMap::new(),
        };

        for block in plan.blocks.iter() {
            for input in plan.handle_inputs.iter() {
                if let Some(input) = input.as_normal()
                    && block.reads.contains_key(&input.relative_id)
                {
                    let analysis =
                        VectorAxisAnalysis::read(block.vector_axis, input.global_ir.shape.rank());

                    actions.record(
                        input.global_ir.id,
                        input.global_ir.id,
                        policy.action(&analysis),
                    );
                }
            }

            for output in plan.handle_outputs.iter() {
                if let HandleOutput::Owned {
                    handle,
                    global_id,
                    relative_id,
                    global_shape,
                    ..
                } = output
                    && block.writes.contains_key(relative_id)
                {
                    let analysis = VectorAxisAnalysis::written(
                        block.vector_axis,
                        global_shape.rank(),
                        &handle.strides,
                    );

                    actions.record(*global_id, *global_id, policy.action(&analysis));
                }
            }
        }

        // A view is vectorized under its own shape and rank.
        // `vectorization_reshape` indexes the reshaped shape at the axis and bails
        // out unless the axis is that shape's last dimension; `vectorization_swapped`
        // maps the axis through the swap and checks the original's stride there — the
        // same mapping [read_input_aligned](crate::engine::codegen::io) applies. Both
        // go wrong only when handed the default while the block iterates elsewhere,
        // and nothing votes for a view's id unless it is done here.
        for view in resources.views.iter() {
            let (view_id, original_id) = match view {
                TensorView::Reshape {
                    reshaped, original, ..
                } => (reshaped, original),
                TensorView::SwapDims {
                    swapped, original, ..
                } => (swapped, original),
                // Already pinned to a vector size of one by the planner.
                TensorView::NhwcStrides { .. } => continue,
            };

            let (Some(view_global), Some(original_global)) = (
                context.tensors.get(view_id),
                context.tensors.get(original_id),
            ) else {
                continue;
            };

            for block in plan.blocks.iter() {
                if !block.reads.contains_key(original_id) {
                    continue;
                }

                let analysis =
                    VectorAxisAnalysis::read(block.vector_axis, view_global.shape.rank());

                actions.record(view_global.id, original_global.id, policy.action(&analysis));
            }
        }

        actions
    }

    fn record(&mut self, id: TensorId, pinned: TensorId, action: VectorAxisAction) {
        self.per_tensor
            .entry(id)
            .and_modify(|(current, _)| *current = current.merged_with(action))
            .or_insert((action, pinned));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_resolved_axis_uses_the_default() {
        let read = VectorAxisAnalysis::read(None, 2);
        assert_eq!(
            VectorAxisPolicy::LineUpWithBlock.action(&read),
            VectorAxisAction::Default
        );
    }

    #[test]
    fn a_read_can_follow_a_nonfinal_axis() {
        let read = VectorAxisAnalysis::read(Some(1), 4);
        assert_eq!(
            VectorAxisPolicy::LineUpWithBlock.action(&read),
            VectorAxisAction::Along(1)
        );
    }

    #[test]
    fn a_read_can_follow_the_last_axis_in_its_rank() {
        let read = VectorAxisAnalysis::read(Some(2), 3);
        assert_eq!(
            VectorAxisPolicy::LineUpWithBlock.action(&read),
            VectorAxisAction::Along(2)
        );
    }

    #[test]
    fn an_axis_outside_the_tensor_rank_is_refused() {
        let read = VectorAxisAnalysis::read(Some(2), 2);
        assert_eq!(
            VectorAxisPolicy::LineUpWithBlock.action(&read),
            VectorAxisAction::Refuse
        );
    }

    #[test]
    fn a_write_along_a_strided_axis_is_refused() {
        let write = VectorAxisAnalysis::written(Some(1), 3, &[12, 4, 1]);
        assert_eq!(
            VectorAxisPolicy::LineUpWithBlock.action(&write),
            VectorAxisAction::Refuse
        );
    }

    #[test]
    fn a_write_along_a_unit_stride_axis_is_allowed() {
        let write = VectorAxisAnalysis::written(Some(1), 3, &[12, 1, 3]);
        assert_eq!(
            VectorAxisPolicy::LineUpWithBlock.action(&write),
            VectorAxisAction::Along(1)
        );
    }

    #[test]
    fn the_default_policy_ignores_a_valid_read_axis() {
        let read = VectorAxisAnalysis::read(Some(1), 4);
        assert_eq!(
            VectorAxisPolicy::AlwaysDefault.action(&read),
            VectorAxisAction::Default
        );
    }

    #[test]
    fn the_default_policy_ignores_an_out_of_rank_read_axis() {
        let read = VectorAxisAnalysis::read(Some(1), 1);
        assert_eq!(
            VectorAxisPolicy::AlwaysDefault.action(&read),
            VectorAxisAction::Default
        );
    }

    #[test]
    fn the_default_policy_ignores_a_strided_write_axis() {
        let write = VectorAxisAnalysis::written(Some(1), 3, &[12, 4, 1]);
        assert_eq!(
            VectorAxisPolicy::AlwaysDefault.action(&write),
            VectorAxisAction::Default
        );
    }

    #[test]
    fn blocks_that_disagree_cancel_out() {
        // Any two actions that are not the same answer refuse the tensor, including
        // a permuted block meeting one that only wants the default.
        assert_eq!(
            VectorAxisAction::Along(1).merged_with(VectorAxisAction::Along(3)),
            VectorAxisAction::Refuse
        );
        assert_eq!(
            VectorAxisAction::Along(1).merged_with(VectorAxisAction::Default),
            VectorAxisAction::Refuse
        );
        assert_eq!(
            VectorAxisAction::Default.merged_with(VectorAxisAction::Default),
            VectorAxisAction::Default
        );
        assert_eq!(
            VectorAxisAction::Along(1).merged_with(VectorAxisAction::Along(1)),
            VectorAxisAction::Along(1)
        );
    }

    #[test]
    fn a_refusal_survives_every_later_action() {
        // Blocks are analysed in whatever order they come in, so a refusal has to
        // be absorbing rather than merely first.
        for other in [
            VectorAxisAction::Default,
            VectorAxisAction::Along(1),
            VectorAxisAction::Refuse,
        ] {
            assert_eq!(
                VectorAxisAction::Refuse.merged_with(other),
                VectorAxisAction::Refuse
            );
            assert_eq!(
                other.merged_with(VectorAxisAction::Refuse),
                VectorAxisAction::Refuse
            );
        }
    }

    #[test]
    fn a_refusal_leaves_a_broadcast_tensor_alone() {
        // Pinning it to `Aligned(1)` instead would make it constrain the block's
        // width, which a broadcast operand does not.
        let broadcast = TensorId::new(0);
        let aligned = TensorId::new(1);
        let refusals = Refusals {
            ids: vec![broadcast, aligned],
        };

        let mut vectorizations =
            BTreeMap::from_iter([(broadcast, Vect::Broadcasted), (aligned, Vect::Aligned(4))]);
        refusals.apply(&mut vectorizations);

        assert!(matches!(vectorizations[&broadcast], Vect::Broadcasted));
        assert!(matches!(vectorizations[&aligned], Vect::Aligned(1)));
    }
}
