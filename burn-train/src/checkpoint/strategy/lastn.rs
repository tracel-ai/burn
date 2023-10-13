use super::CheckpointingStrategy;
use crate::{checkpoint::CheckpointingAction, EventCollector};

/// Keep the last N checkpoints.
///
/// Very useful when training, minimizing disk space while ensuring that the training can be
/// resumed even if something goes wrong.
#[derive(new)]
pub struct KeepLastNCheckpoints {
    num_keep: usize,
}

impl<E: EventCollector> CheckpointingStrategy<E> for KeepLastNCheckpoints {
    fn checkpointing(&mut self, epoch: usize, _collector: &mut E) -> Vec<CheckpointingAction> {
        let mut actions = vec![CheckpointingAction::Save];

        if let Some(epoch) = usize::checked_sub(epoch, self.num_keep) {
            if epoch > 0 {
                actions.push(CheckpointingAction::Delete(epoch));
            }
        }

        actions
    }
}

#[cfg(test)]
mod tests {
    use crate::{info::MetricsInfo, test_utils::TestEventCollector};

    use super::*;

    #[test]
    fn should_always_delete_lastn_epoch_if_higher_than_one() {
        let mut strategy = KeepLastNCheckpoints::new(2);
        let mut collector = TestEventCollector::<f64, f64>::new(MetricsInfo::new());

        assert_eq!(
            vec![CheckpointingAction::Save],
            strategy.checkpointing(1, &mut collector)
        );

        assert_eq!(
            vec![CheckpointingAction::Save],
            strategy.checkpointing(2, &mut collector)
        );

        assert_eq!(
            vec![CheckpointingAction::Save, CheckpointingAction::Delete(1)],
            strategy.checkpointing(3, &mut collector)
        );
    }
}
