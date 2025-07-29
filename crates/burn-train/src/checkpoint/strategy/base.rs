use std::ops::DerefMut;

use crate::metric::store::EventStoreClient;

/// Action to be taken by a [checkpointer](crate::checkpoint::Checkpointer).
#[derive(Clone, PartialEq, Debug)]
pub enum CheckpointingAction {
    /// Delete the given epoch.
    Delete(usize),
    /// Save the current record.
    Save,
}

/// Define when checkpoint should be saved and deleted.
pub trait CheckpointingStrategy: Send {
    /// Based on the epoch, determine if the checkpoint should be saved.
    fn checkpointing(
        &mut self,
        epoch: usize,
        collector: &EventStoreClient,
    ) -> Vec<CheckpointingAction>;
}

// We make dyn box implement the checkpointing strategy so that it can be used with generic, but
// still be dynamic.
impl CheckpointingStrategy for Box<dyn CheckpointingStrategy> {
    fn checkpointing(
        &mut self,
        epoch: usize,
        collector: &EventStoreClient,
    ) -> Vec<CheckpointingAction> {
        self.deref_mut().checkpointing(epoch, collector)
    }
}
