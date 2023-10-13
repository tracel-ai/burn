use super::{CheckpointingAction, CheckpointingStrategy};
use crate::EventCollector;
use std::collections::HashSet;

/// Compose multiple checkpointing strategy and only delete checkpoints when both strategy flag an
/// epoch to be deleted.
pub struct ComposedCheckpointingStrategy<E: EventCollector> {
    strategies: Vec<Box<dyn CheckpointingStrategy<E>>>,
    deleted: Vec<HashSet<usize>>,
}

impl<E: EventCollector> ComposedCheckpointingStrategy<E> {
    /// Create a new composed checkpointing strategy.
    pub fn new(strategies: Vec<Box<dyn CheckpointingStrategy<E>>>) -> Self {
        Self {
            deleted: strategies.iter().map(|_| HashSet::new()).collect(),
            strategies,
        }
    }
}

impl<E: EventCollector> CheckpointingStrategy<E> for ComposedCheckpointingStrategy<E> {
    fn checkpointing(&mut self, epoch: usize, collector: &mut E) -> Vec<CheckpointingAction> {
        let mut saved = false;
        let mut actions = Vec::new();
        let mut epoch_to_check = Vec::new();

        for (i, strategy) in self.strategies.iter_mut().enumerate() {
            let actions = strategy.checkpointing(epoch, collector);
            // We assume that the strategy would not want the current epoch to be saved.
            // So we flag it as deleted.
            if actions.is_empty() {
                self.deleted.get_mut(i).unwrap().insert(epoch);
            }

            for action in actions {
                match action {
                    CheckpointingAction::Delete(epoch) => {
                        self.deleted.get_mut(i).unwrap().insert(epoch);
                        epoch_to_check.push(epoch);
                    }
                    CheckpointingAction::Save => saved = true,
                }
            }
        }

        if saved {
            actions.push(CheckpointingAction::Save);
        }

        for epoch in epoch_to_check.into_iter() {
            let mut num_true = 0;
            for i in 0..self.strategies.len() {
                if self.deleted.get(i).unwrap().contains(&epoch) {
                    num_true += 1;
                }
            }

            if num_true == self.strategies.len() {
                actions.push(CheckpointingAction::Delete(epoch));

                for i in 0..self.strategies.len() {
                    self.deleted.get_mut(i).unwrap().remove(&epoch);
                }
            }
        }

        actions
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        checkpoint::KeepLastNCheckpoints, info::MetricsInfo, test_utils::TestEventCollector,
    };

    use super::*;

    #[test]
    fn should_delete_when_both_deletes() {
        let strategy_1 = KeepLastNCheckpoints::new(1);
        let strategy_2 = KeepLastNCheckpoints::new(2);
        let mut collector = TestEventCollector::<f64, f64>::new(MetricsInfo::new());
        let mut strategy =
            ComposedCheckpointingStrategy::new(vec![Box::new(strategy_1), Box::new(strategy_2)]);

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
