use burn_core::{self as burn, module::ParamId};

use burn_core::record::Record;

use crate::group::{ParamGroup, ParamGroupRecord};
use crate::{
    LearningRate,
    lr_scheduler::{
        base::LrScheduler,
        composed::{LrSchedulerItem, LrSchedulerRecord},
    },
};

/// Regroups multiple learning rates.
#[derive(Clone)]
pub struct MultiLearningRate {
    /// The default learning rate.
    pub default: LearningRate,
    /// [ParamGroup]'s for different learning rates.
    pub groups: Vec<ParamGroup<LearningRate>>,
}

/// A Record for a [MultiLrScheduler].
#[derive(Clone, Record)]
pub struct MultiLrSchedulerRecord {
    /// The record for the default scheduler.
    pub default: LrSchedulerRecord,
    /// The record for the groups.
    pub groups: Vec<ParamGroupRecord<LrSchedulerRecord>>,
}

/// Regroups multiple [LrSchedulerItem]'s.
#[derive(Clone)]
pub struct MultiLrScheduler {
    /// The default scheduler.
    pub default: LrSchedulerItem,
    /// [ParamGroup]'s for different schedulers.
    pub schedulers: Vec<ParamGroup<LrSchedulerItem>>,
}

impl MultiLrScheduler {
    /// Performs a step for all schedulers and returns the effective learning rates.
    pub fn step(&mut self) -> MultiLearningRate {
        let default = match &mut self.default {
            LrSchedulerItem::Linear(s) => s.step(),
            LrSchedulerItem::Cosine(s) => s.step(),
            LrSchedulerItem::Exponential(s) => s.step(),
            LrSchedulerItem::Noam(s) => s.step(),
        };

        let groups = self
            .schedulers
            .iter_mut()
            .map(|group| {
                let lr = match &mut group.config {
                    LrSchedulerItem::Linear(s) => s.step(),
                    LrSchedulerItem::Cosine(s) => s.step(),
                    LrSchedulerItem::Exponential(s) => s.step(),
                    LrSchedulerItem::Noam(s) => s.step(),
                };
                ParamGroup::new(group.tag.clone(), group.params.clone(), lr)
            })
            .collect();

        MultiLearningRate { default, groups }
    }

    /// Get the current states of the schedulers as a [record](Record).
    pub fn to_record(&self) -> MultiLrSchedulerRecord {
        let default = match &self.default {
            LrSchedulerItem::Linear(item) => LrSchedulerRecord::Linear(item.to_record()),
            LrSchedulerItem::Cosine(item) => LrSchedulerRecord::Cosine(item.to_record()),
            LrSchedulerItem::Exponential(item) => LrSchedulerRecord::Exponential(item.to_record()),
            LrSchedulerItem::Noam(item) => LrSchedulerRecord::Noam(item.to_record()),
        };

        let groups = self
            .schedulers
            .iter()
            .map(|s| {
                let record = match &s.config {
                    LrSchedulerItem::Linear(item) => LrSchedulerRecord::Linear(item.to_record()),
                    LrSchedulerItem::Cosine(item) => LrSchedulerRecord::Cosine(item.to_record()),
                    LrSchedulerItem::Exponential(item) => {
                        LrSchedulerRecord::Exponential(item.to_record())
                    }
                    LrSchedulerItem::Noam(item) => LrSchedulerRecord::Noam(item.to_record()),
                };
                let params = s.params.iter().map(|p| p.clone().into()).collect();
                ParamGroupRecord::new(s.tag.clone(), params, record)
            })
            .collect();

        MultiLrSchedulerRecord { default, groups }
    }

    /// Load the states of the schedulers as a [record](Record).
    pub fn load_record(self, record: MultiLrSchedulerRecord) -> Self {
        let default = match (self.default, record.default) {
            (LrSchedulerItem::Linear(item), LrSchedulerRecord::Linear(record)) => {
                LrSchedulerItem::Linear(item.load_record(record))
            }
            (LrSchedulerItem::Cosine(item), LrSchedulerRecord::Cosine(record)) => {
                LrSchedulerItem::Cosine(item.load_record(record))
            }
            (LrSchedulerItem::Exponential(item), LrSchedulerRecord::Exponential(record)) => {
                LrSchedulerItem::Exponential(item.load_record(record))
            }
            (LrSchedulerItem::Noam(item), LrSchedulerRecord::Noam(record)) => {
                LrSchedulerItem::Noam(item.load_record(record))
            }
            _ => panic!("Invalid state"),
        };

        let schedulers = self
            .schedulers
            .into_iter()
            .zip(record.groups)
            .map(|(scheduler, record)| {
                let config = match (scheduler.config, record.config) {
                    (LrSchedulerItem::Linear(item), LrSchedulerRecord::Linear(record)) => {
                        LrSchedulerItem::Linear(item.load_record(record))
                    }
                    (LrSchedulerItem::Cosine(item), LrSchedulerRecord::Cosine(record)) => {
                        LrSchedulerItem::Cosine(item.load_record(record))
                    }
                    (
                        LrSchedulerItem::Exponential(item),
                        LrSchedulerRecord::Exponential(record),
                    ) => LrSchedulerItem::Exponential(item.load_record(record)),
                    (LrSchedulerItem::Noam(item), LrSchedulerRecord::Noam(record)) => {
                        LrSchedulerItem::Noam(item.load_record(record))
                    }
                    _ => panic!("Invalid state"),
                };
                let params = record
                    .params
                    .into_iter()
                    .map(|p| ParamId::from(p.value))
                    .collect();
                ParamGroup::new(record.tag, params, config)
            })
            .collect();

        MultiLrScheduler {
            default,
            schedulers,
        }
    }
}
