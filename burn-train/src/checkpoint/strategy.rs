use crate::{
    metric::{LossMetric, Metric},
    Aggregate, Direction, EventCollector, Split,
};
use std::ops::DerefMut;

pub enum CheckpointerAction {
    Delete(usize),
    Save,
}

/// Define when checkpoint should be saved and deleted.
pub trait CheckpointerStrategy<E: EventCollector> {
    /// Based on the epoch, determine if the checkpoint should be saved.
    fn checkpointing(&mut self, epoch: usize, collector: &mut E) -> Vec<CheckpointerAction>;
}

impl<E: EventCollector> CheckpointerStrategy<E> for Box<dyn CheckpointerStrategy<E>> {
    fn checkpointing(&mut self, epoch: usize, collector: &mut E) -> Vec<CheckpointerAction> {
        self.deref_mut().checkpointing(epoch, collector)
    }
}

#[derive(new)]
pub struct KeepLastN {
    num_keep: usize,
}

impl<E: EventCollector> CheckpointerStrategy<E> for KeepLastN {
    fn checkpointing(&mut self, epoch: usize, _collector: &mut E) -> Vec<CheckpointerAction> {
        vec![
            CheckpointerAction::Save,
            CheckpointerAction::Delete(epoch - self.num_keep),
        ]
    }
}

#[derive(new)]
pub struct LowestLoss {
    current: Option<usize>,
}

impl<E: EventCollector> CheckpointerStrategy<E> for LowestLoss {
    fn checkpointing(&mut self, epoch: usize, collector: &mut E) -> Vec<CheckpointerAction> {
        if let Some(epoch_best) =
            collector.find_epoch("Loss", Aggregate::Mean, Direction::Lowest, Split::Valid)
        {
            if let Some(epoch_current) = self.current.clone() {
                if epoch_current == epoch_best {
                    return vec![CheckpointerAction::Save];
                } else {
                    self.current = Some(epoch_best);

                    return vec![
                        CheckpointerAction::Save,
                        CheckpointerAction::Delete(epoch_current),
                    ];
                }
            } else {
                self.current = Some(epoch_best);
                return vec![CheckpointerAction::Save];
            }
        }

        self.current = Some(epoch);
        vec![CheckpointerAction::Save]
    }
}
