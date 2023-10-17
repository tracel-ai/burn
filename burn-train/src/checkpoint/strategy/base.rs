use crate::EventCollector;
use std::ops::DerefMut;

/// Action to be taken by a [checkpointer](crate::checkpoint::Checkpointer).
#[derive(Clone, PartialEq, Debug)]
pub enum CheckpointingAction {
    /// Delete the given epoch.
    Delete(usize),
    /// Save the current record.
    Save,
}

/// Define when checkpoint should be saved and deleted.
pub trait CheckpointingStrategy<E: EventCollector> {
    /// Based on the epoch, determine if the checkpoint should be saved.
    fn checkpointing(&mut self, epoch: usize, collector: &mut E) -> Vec<CheckpointingAction>;
}

// We make dyn box implement the checkpointing strategy so that it can be used with generic, but
// still be dynamic.
impl<E: EventCollector> CheckpointingStrategy<E> for Box<dyn CheckpointingStrategy<E>> {
    fn checkpointing(&mut self, epoch: usize, collector: &mut E) -> Vec<CheckpointingAction> {
        self.deref_mut().checkpointing(epoch, collector)
    }
}
