use core::fmt::Debug;

use burn_backend::Backend;

use crate::{graph::ComputingProperty, tensor::AutodiffTensor};
use alloc::sync::Arc;

use super::{
    builder::{ActionType, CheckpointerBuilder},
    retro_forward::RetroForward,
};

/// Strategy for the amount of checkpointing to do during autodiff
pub trait CheckpointStrategy: Clone + Copy + Debug + Default + Send + Sync + 'static {
    /// May modify the compute property depending on the strategy
    fn compute_property<R: RetroForward>(retro_forward: R) -> ComputingProperty;

    /// Checkpoints parents if necessary in the strategy
    fn checkpoint_parents<'a, B2, A>(
        parents: A,
        builder: &mut CheckpointerBuilder,
    ) -> Result<(), CheckpointingError>
    where
        B2: Backend,
        A: IntoIterator<Item = &'a AutodiffTensor<B2>>;
}

#[derive(Debug)]
/// Error that can happen when trying to checkpoint a tensor.
pub enum CheckpointingError {
    /// When a parent is untracked, we can't easily checkpoint its state, since we don't know the
    /// requirements in advanced.
    UntrackedParent,
}

#[derive(Clone, Copy, Debug, Default)]
/// All operations are considered compute bound, notwithstanding how they are marked
pub struct NoCheckpointing {}

impl CheckpointStrategy for NoCheckpointing {
    /// An operation marked as memory bound is actually compute bound.
    fn compute_property<R: RetroForward>(_retro_forward: R) -> ComputingProperty {
        ComputingProperty::ComputeBound
    }

    /// An operation marked as memory bound is actually compute bound.
    /// It's therefore useless to checkpoint the parents
    fn checkpoint_parents<'a, B2, A>(
        _parents: A,
        _builder: &mut CheckpointerBuilder,
    ) -> Result<(), CheckpointingError>
    where
        B2: Backend,
        A: IntoIterator<Item = &'a AutodiffTensor<B2>>,
    {
        // Nothing to do here
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default)]
/// Operation properties are as they are marked (compute or memory bound)
pub struct BalancedCheckpointing {}

impl CheckpointStrategy for BalancedCheckpointing {
    /// An operation marked as memory bound is memory bound.
    /// When memory bound, an operation needs to save its RetroForward
    fn compute_property<R: RetroForward>(retro_forward: R) -> ComputingProperty {
        ComputingProperty::MemoryBound {
            retro_forward: Arc::new(retro_forward),
        }
    }

    /// An operation marked as memory bound is really memory bound.
    /// Since the operation may not checkpoint its parents but may need them indirectly
    /// if asked to recompute itself, the method needs to know the parent tensors to maybe checkpoint them
    fn checkpoint_parents<'a, B2, A>(
        parents: A,
        builder: &mut CheckpointerBuilder,
    ) -> Result<(), CheckpointingError>
    where
        B2: Backend,
        A: IntoIterator<Item = &'a AutodiffTensor<B2>>,
    {
        let mut can_checkpoint = true;

        for tensor in parents.into_iter() {
            if let crate::graph::Requirement::None = tensor.node.requirement {
                can_checkpoint = false;
            } else {
                builder.checkpoint(tensor, ActionType::Backup);
            }
        }

        if !can_checkpoint {
            *builder = CheckpointerBuilder::default();
            return Err(CheckpointingError::UntrackedParent);
        }

        Ok(())
    }
}
