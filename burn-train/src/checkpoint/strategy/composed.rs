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
            for action in strategy.checkpointing(epoch, collector) {
                match action {
                    CheckpointingAction::Delete(epoch) => {
                        self.deleted.get_mut(i).unwrap().insert(epoch);
                        epoch_to_check.push(epoch);
                    }
                    CheckpointingAction::Save => saved = true,
                }
            }
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

        if saved {
            actions.push(CheckpointingAction::Save);
        }

        actions
    }
}
