use super::CheckpointingStrategy;
use crate::{checkpoint::CheckpointingAction, metric::store::EventStoreClient};

/// Keep the last N checkpoints.
///
/// Very useful when training, minimizing disk space while ensuring that the training can be
/// resumed even if something goes wrong.
#[derive(new)]
pub struct KeepLastNCheckpoints {
    num_keep: usize,
}

impl CheckpointingStrategy for KeepLastNCheckpoints {
    fn checkpointing(
        &mut self,
        epoch: usize,
        _store: &EventStoreClient,
    ) -> Vec<CheckpointingAction> {
        let mut actions = vec![CheckpointingAction::Save];

        if let Some(epoch) = usize::checked_sub(epoch, self.num_keep)
            && epoch > 0
        {
            actions.push(CheckpointingAction::Delete(epoch));
        }

        actions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::store::LogEventStore;

    #[test]
    fn should_always_delete_lastn_epoch_if_higher_than_one() {
        let mut strategy = KeepLastNCheckpoints::new(2);
        let store = EventStoreClient::new(LogEventStore::default());

        assert_eq!(
            vec![CheckpointingAction::Save],
            strategy.checkpointing(1, &store)
        );

        assert_eq!(
            vec![CheckpointingAction::Save],
            strategy.checkpointing(2, &store)
        );

        assert_eq!(
            vec![CheckpointingAction::Save, CheckpointingAction::Delete(1)],
            strategy.checkpointing(3, &store)
        );
    }
}
