use crate::{metric::Metric, Aggregate, Direction, EventCollector, Split};
use std::{collections::HashSet, ops::DerefMut};

/// Action to be taken by a [checkpointer](crate::checkpoint::Checkpointer).
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
        vec![
            CheckpointingAction::Save,
            CheckpointingAction::Delete(epoch - self.num_keep),
        ]
    }
}

/// Keep the best checkpoint based on a metric.
pub struct MetricCheckpointingStrategy {
    current: Option<usize>,
    aggregate: Aggregate,
    direction: Direction,
    split: Split,
    name: String,
}

impl MetricCheckpointingStrategy {
    /// Create a new metric strategy.
    pub fn new<M>(aggregate: Aggregate, direction: Direction, split: Split) -> Self
    where
        M: Metric,
    {
        Self {
            current: None,
            name: M::NAME.to_string(),
            aggregate,
            direction,
            split,
        }
    }
}

impl<E: EventCollector> CheckpointingStrategy<E> for MetricCheckpointingStrategy {
    fn checkpointing(&mut self, epoch: usize, collector: &mut E) -> Vec<CheckpointingAction> {
        let best_epoch =
            match collector.find_epoch(&self.name, self.aggregate, self.direction, self.split) {
                Some(epoch_best) => epoch_best,
                None => epoch,
            };

        let mut actions = Vec::new();

        if let Some(current) = self.current {
            if current != best_epoch {
                actions.push(CheckpointingAction::Delete(current));
            }
        }

        if best_epoch == epoch {
            actions.push(CheckpointingAction::Save);
        }

        self.current = Some(best_epoch);

        actions
    }
}

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
