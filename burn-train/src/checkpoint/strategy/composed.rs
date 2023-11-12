use crate::metric::store::EventStoreClient;

use super::{CheckpointingAction, CheckpointingStrategy};
use std::collections::HashSet;

/// Compose multiple checkpointing strategy and only delete checkpoints when both strategy flag an
/// epoch to be deleted.
pub struct ComposedCheckpointingStrategy {
    strategies: Vec<Box<dyn CheckpointingStrategy>>,
    deleted: Vec<HashSet<usize>>,
}

/// Help building a [checkpointing strategy](CheckpointingStrategy) by combining multiple ones.
#[derive(Default)]
pub struct ComposedCheckpointingStrategyBuilder {
    strategies: Vec<Box<dyn CheckpointingStrategy>>,
}

impl ComposedCheckpointingStrategyBuilder {
    /// Add a new [checkpointing strategy](CheckpointingStrategy).
    #[allow(clippy::should_implement_trait)]
    pub fn add<S>(mut self, strategy: S) -> Self
    where
        S: CheckpointingStrategy + 'static,
    {
        self.strategies.push(Box::new(strategy));
        self
    }

    /// Create a new [composed checkpointing strategy](ComposedCheckpointingStrategy).
    pub fn build(self) -> ComposedCheckpointingStrategy {
        ComposedCheckpointingStrategy::new(self.strategies)
    }
}

impl ComposedCheckpointingStrategy {
    fn new(strategies: Vec<Box<dyn CheckpointingStrategy>>) -> Self {
        Self {
            deleted: strategies.iter().map(|_| HashSet::new()).collect(),
            strategies,
        }
    }
    /// Create a new builder which help compose multiple
    /// [checkpointing strategies](CheckpointingStrategy).
    pub fn builder() -> ComposedCheckpointingStrategyBuilder {
        ComposedCheckpointingStrategyBuilder::default()
    }
}

impl CheckpointingStrategy for ComposedCheckpointingStrategy {
    fn checkpointing(
        &mut self,
        epoch: usize,
        collector: &EventStoreClient,
    ) -> Vec<CheckpointingAction> {
        let mut saved = false;
        let mut actions = Vec::new();
        let mut epochs_to_check = Vec::new();

        for (i, strategy) in self.strategies.iter_mut().enumerate() {
            let actions = strategy.checkpointing(epoch, collector);
            // We assume that the strategy would not want the current epoch to be saved.
            // So we flag it as deleted.
            if actions.is_empty() {
                self.deleted
                    .get_mut(i)
                    .expect("As many 'deleted' as 'strategies'.")
                    .insert(epoch);
            }

            for action in actions {
                match action {
                    CheckpointingAction::Delete(epoch) => {
                        self.deleted
                            .get_mut(i)
                            .expect("As many 'deleted' as 'strategies'.")
                            .insert(epoch);
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
                if self
                    .deleted
                    .get(i)
                    .expect("Ad many 'deleted' as 'strategies'.")
                    .contains(&epoch)
                {
                    num_true += 1;
                }
            }

            if num_true == self.strategies.len() {
                actions.push(CheckpointingAction::Delete(epoch));

                for i in 0..self.strategies.len() {
                    self.deleted
                        .get_mut(i)
                        .expect("As many 'deleted' as 'strategies'.")
                        .remove(&epoch);
                }
            }
        }

        actions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{checkpoint::KeepLastNCheckpoints, metric::store::LogEventStore};

    #[test]
    fn should_delete_when_both_deletes() {
        let store = EventStoreClient::new(LogEventStore::default());
        let mut strategy = ComposedCheckpointingStrategy::builder()
            .add(KeepLastNCheckpoints::new(1))
            .add(KeepLastNCheckpoints::new(2))
            .build();

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
