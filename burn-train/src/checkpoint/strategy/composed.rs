use super::{CheckpointingAction, CheckpointingStrategy};
use crate::EventCollector;
use std::collections::HashSet;

/// Compose multiple checkpointing strategy and only delete checkpoints when both strategy flag an
/// epoch to be deleted.
pub struct ComposedCheckpointingStrategy<E: EventCollector> {
    strategies: Vec<Box<dyn CheckpointingStrategy<E>>>,
    deleted: Vec<HashSet<usize>>,
}

/// Help building a [checkpointing strategy](CheckpointingStrategy) by combining multiple ones.
pub struct ComposedCheckpointingStrategyBuilder<E: EventCollector> {
    strategies: Vec<Box<dyn CheckpointingStrategy<E>>>,
}

impl<E: EventCollector> Default for ComposedCheckpointingStrategyBuilder<E> {
    fn default() -> Self {
        Self {
            strategies: Vec::new(),
        }
    }
}

impl<E: EventCollector> ComposedCheckpointingStrategyBuilder<E> {
    /// Add a new [checkpointing strategy](CheckpointingStrategy).
    pub fn add<S>(mut self, strategy: S) -> Self
    where
        S: CheckpointingStrategy<E> + 'static,
    {
        self.strategies.push(Box::new(strategy));
        self
    }

    /// Create a new [composed checkpointing strategy](ComposedCheckpointingStrategy).
    pub fn build(self) -> ComposedCheckpointingStrategy<E> {
        ComposedCheckpointingStrategy::new(self.strategies)
    }
}

impl<E: EventCollector> ComposedCheckpointingStrategy<E> {
    fn new(strategies: Vec<Box<dyn CheckpointingStrategy<E>>>) -> Self {
        Self {
            deleted: strategies.iter().map(|_| HashSet::new()).collect(),
            strategies,
        }
    }
    /// Create a new builder which help compose multiple
    /// [checkpointing strategies](CheckpointingStrategy).
    pub fn builder() -> ComposedCheckpointingStrategyBuilder<E> {
        ComposedCheckpointingStrategyBuilder::default()
    }
}

impl<E: EventCollector> CheckpointingStrategy<E> for ComposedCheckpointingStrategy<E> {
    fn checkpointing(&mut self, epoch: usize, collector: &mut E) -> Vec<CheckpointingAction> {
        let mut saved = false;
        let mut actions = Vec::new();
        let mut epochs_to_check = Vec::new();

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
                        epochs_to_check.push(epoch);
                    }
                    CheckpointingAction::Save => saved = true,
                }
            }
        }

        if saved {
            actions.push(CheckpointingAction::Save);
        }

        for epoch in epochs_to_check.into_iter() {
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
        let mut collector = TestEventCollector::<f64, f64>::new(MetricsInfo::new());
        let mut strategy = ComposedCheckpointingStrategy::builder()
            .add(KeepLastNCheckpoints::new(1))
            .add(KeepLastNCheckpoints::new(2))
            .build();

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
