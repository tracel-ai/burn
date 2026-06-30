use burn_core::{self as burn, module::ParamId};

use burn::config::Config;
use burn_core::module::ParamGroup;

use crate::{
    LearningRate,
    lr_scheduler::{DynLrScheduler, LrScheduler, LrSchedulerConfig, LrSchedulerRecord},
};

#[derive(Clone)]
struct LrGroup {
    group: ParamGroup,
    lr: f64,
}

/// A policy that determines what learning rate to use for a trainable parameter.
#[derive(Clone, Default)]
pub struct LrPolicy {
    default: f64,
    groups: Vec<LrGroup>,
}

impl From<LearningRate> for LrPolicy {
    fn from(value: LearningRate) -> Self {
        Self {
            default: value,
            groups: vec![],
        }
    }
}

impl LrPolicy {
    /// Get the effective learning rate for the given parameter.
    pub fn lr_from_param(&self, id: ParamId, path: Option<&str>) -> LearningRate {
        self.groups
            .iter()
            .filter_map(|val| {
                val.group
                    .matches(&id, path)
                    .expect("Failed to match a parameter group.")
                    .then_some(val.lr)
            })
            .last()
            .unwrap_or(self.default)
    }

    /// Get the default learning rate value.
    pub fn default(&self) -> LearningRate {
        self.default
    }
}

#[derive(Config, Debug)]
struct LrSchedulerGroupConfig {
    group: ParamGroup,
    scheduler: LrSchedulerConfig,
}

#[derive(new, Clone)]
struct LrSchedulerGroup {
    group: ParamGroup,
    scheduler: DynLrScheduler,
}

/// Configuration for a [LrPolicyScheduler].
#[derive(Config, Debug)]
pub struct LrPolicyConfig {
    default: LrSchedulerConfig,
    #[config(default = "Vec::new()")]
    scheduler_groups: Vec<LrSchedulerGroupConfig>,
}

/// A learning rate scheduler that maps specific parameter groups to dedicated sub-schedulers,
/// falling back to a global default scheduler for unmapped parameters.
///
/// This allows heterogeneous learning rate schedules across different layers of a model
/// (e.g., discriminative layer training or fine-tuning).
#[derive(Clone)]
pub struct LrPolicyScheduler {
    default: DynLrScheduler,
    scheduler_groups: Vec<LrSchedulerGroup>,
}

impl LrPolicyConfig {
    /// Initialize a new learning rate policy scheduler.
    pub fn init(&self) -> Result<LrPolicyScheduler, String> {
        let mut groups = Vec::with_capacity(self.scheduler_groups.len());
        for group in self.scheduler_groups.iter() {
            let scheduler = match &group.scheduler {
                LrSchedulerConfig::Linear(config) => config.init()?.into(),
                LrSchedulerConfig::Cosine(config) => config.init()?.into(),
                LrSchedulerConfig::Exponential(config) => config.init()?.into(),
                LrSchedulerConfig::Noam(config) => config.init()?.into(),
                LrSchedulerConfig::Step(config) => config.init()?.into(),
                LrSchedulerConfig::Composed(config) => config.init()?.into(),
            };
            groups.push(LrSchedulerGroup::new(group.group.clone(), scheduler));
        }

        let default = match &self.default {
            LrSchedulerConfig::Linear(config) => config.init()?.into(),
            LrSchedulerConfig::Cosine(config) => config.init()?.into(),
            LrSchedulerConfig::Exponential(config) => config.init()?.into(),
            LrSchedulerConfig::Noam(config) => config.init()?.into(),
            LrSchedulerConfig::Step(config) => config.init()?.into(),
            LrSchedulerConfig::Composed(config) => config.init()?.into(),
        };

        Ok(LrPolicyScheduler {
            default,
            scheduler_groups: groups,
        })
    }

    /// Set the default learning rate scheduler.
    pub fn with_default_scheduler(mut self, scheduler: LrSchedulerConfig) -> Self {
        self.default = scheduler;
        self
    }

    /// Add a new parameter group to the scheduler's policy.
    pub fn add_group(mut self, group: ParamGroup, scheduler: LrSchedulerConfig) -> Self {
        self.scheduler_groups
            .push(LrSchedulerGroupConfig { group, scheduler });
        self
    }
}

impl LrPolicyScheduler {
    /// Create a [LrPolicyScheduler] with no registered parameter group.
    ///
    /// # Arguments
    ///
    /// * `default_scheduler` - The policy's default learning rate scheduler.
    pub fn new<S: LrScheduler + 'static>(default_scheduler: S) -> Self {
        Self {
            default: default_scheduler.into(),
            scheduler_groups: vec![],
        }
    }

    /// Perform the scheduler step of every scheduler and returns the effective learning rate policy.
    pub fn step(&mut self) -> LrPolicy {
        let default_lr = self.default.step();

        let groups = self
            .scheduler_groups
            .iter_mut()
            .map(|s| {
                let lr = s.scheduler.step();

                LrGroup {
                    group: s.group.clone(),
                    lr,
                }
            })
            .collect();

        LrPolicy {
            default: default_lr,
            groups,
        }
    }

    // TODO: should the param group be saved/loaded along with the record?
    /// Get the current state of the schedulers as a [record](LrSchedulerRecord).
    pub fn to_record(&self) -> super::LrSchedulerRecord {
        let mut record = LrSchedulerRecord::new();
        for (index, item) in self.scheduler_groups.iter().enumerate() {
            let sub = item.scheduler.to_record();
            // record = record.with_record(&index.to_string(), sub.with_group(item.group.clone()));
            record = record.with_record(&index.to_string(), sub);
        }

        let default_record = self.default.to_record();
        record = record.with_record(&self.scheduler_groups.len().to_string(), default_record);

        record
    }

    // TODO: should the param group be saved/loaded along with the record?
    /// Load the state of the schedulers from a [record](LrSchedulerRecord).
    pub fn load_record(mut self, record: super::LrSchedulerRecord) -> Self {
        self.scheduler_groups = self
            .scheduler_groups
            .into_iter()
            .enumerate()
            .map(|(index, item)| {
                let sub = record.record(&index.to_string());
                let scheduler = item.scheduler.load_record(sub);

                LrSchedulerGroup {
                    group: item.group,
                    scheduler,
                }
            })
            .collect();

        let sub = record.record(&self.scheduler_groups.len().to_string());
        self.default = self.default.load_record(sub);

        self
    }
}
