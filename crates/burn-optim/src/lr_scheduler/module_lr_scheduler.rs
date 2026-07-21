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

/// Determines what learning rate to use for a given trainable parameter.
#[derive(Clone, Default)]
pub struct ModuleLearningRate {
    groups: Vec<LrGroup>,
}

impl From<LearningRate> for ModuleLearningRate {
    fn from(value: LearningRate) -> Self {
        Self {
            groups: vec![LrGroup {
                group: ParamGroup::all(),
                lr: value,
            }],
        }
    }
}

impl ModuleLearningRate {
    /// Get the effective learning rate for the given parameter.
    pub fn lr_from_param(&self, id: ParamId, path: Option<&str>) -> LearningRate {
        self.groups
            .iter()
            .filter_map(|val| val.group.matches(&id, path).then_some(val.lr))
            .next_back()
            .expect("Should match at least one parameter group.")
    }

    /// Get the base learning rate value which's group matches all parameters.
    pub fn base(&self) -> LearningRate {
        self.groups
            .first()
            .expect("Should have at least one learning rate.")
            .lr
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

/// Configuration for a [ModuleLrScheduler].
#[derive(Config, Debug)]
pub struct ModuleLrSchedulerConfig {
    base: LrSchedulerConfig,
    #[config(default = "Vec::new()")]
    scheduler_groups: Vec<LrSchedulerGroupConfig>,
}

/// A learning rate scheduler that maps specific parameter groups to dedicated sub-schedulers.
///
/// This allows heterogeneous learning rate schedules across different layers of a model
/// (e.g., discriminative layer training or fine-tuning).
#[derive(Clone)]
pub struct ModuleLrScheduler {
    groups: Vec<LrSchedulerGroup>,
}

impl ModuleLrSchedulerConfig {
    /// Initialize a new learning rate policy scheduler.
    pub fn init(&self) -> Result<ModuleLrScheduler, String> {
        let mut groups = Vec::with_capacity(self.scheduler_groups.len());

        let base = self.base.build()?;
        groups.push(LrSchedulerGroup {
            group: ParamGroup::all(),
            scheduler: base,
        });

        for group in self.scheduler_groups.iter() {
            let scheduler = group.scheduler.build()?;
            groups.push(LrSchedulerGroup::new(group.group.clone(), scheduler));
        }

        Ok(ModuleLrScheduler { groups })
    }

    /// Add a new parameter group to the scheduler's policy.
    pub fn with_group(
        mut self,
        group: ParamGroup,
        scheduler: impl Into<LrSchedulerConfig>,
    ) -> Self {
        self.scheduler_groups.push(LrSchedulerGroupConfig {
            group,
            scheduler: scheduler.into(),
        });
        self
    }
}

impl ModuleLrScheduler {
    /// Create a [ModuleLrScheduler].
    ///
    /// # Arguments
    ///
    /// * `scheduler` - The policy's default learning rate scheduler.
    pub fn new<S: LrScheduler + 'static>(scheduler: S) -> Self {
        Self {
            groups: vec![LrSchedulerGroup {
                group: ParamGroup::all(),
                scheduler: scheduler.into(),
            }],
        }
    }

    /// Perform the scheduler step of every scheduler and returns the effective learning rate policy.
    pub fn step(&mut self) -> ModuleLearningRate {
        let groups = self
            .groups
            .iter_mut()
            .map(|s| {
                let lr = s.scheduler.step();

                LrGroup {
                    group: s.group.clone(),
                    lr,
                }
            })
            .collect();

        ModuleLearningRate { groups }
    }

    /// Get the current state of the schedulers as a [record](LrSchedulerRecord).
    pub fn to_record(&self) -> super::LrSchedulerRecord {
        let mut record = LrSchedulerRecord::new();
        for (index, item) in self.groups.iter().enumerate() {
            let sub = item.scheduler.to_record();
            record = record.with_record(&index.to_string(), sub);
        }

        record
    }

    /// Load the state of the schedulers from a [record](LrSchedulerRecord).
    pub fn load_record(mut self, record: super::LrSchedulerRecord) -> Self {
        self.groups = self
            .groups
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

        self
    }

    /// Add a new parameter group to the scheduler's policy.
    pub fn with_group(mut self, group: ParamGroup, scheduler: impl Into<DynLrScheduler>) -> Self {
        self.groups.push(LrSchedulerGroup {
            group,
            scheduler: scheduler.into(),
        });
        self
    }
}

impl<S> From<S> for ModuleLrScheduler
where
    S: LrScheduler + 'static,
{
    fn from(value: S) -> Self {
        Self::new(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lr_scheduler::linear::LinearLrSchedulerConfig;
    use burn_core::module::ParamGroup;

    const EPSILON: f64 = 1e-10;

    fn check_approx(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < EPSILON,
            "expected {expected}, got {actual}",
        );
    }

    #[test]
    fn step_yields_constant_default_lr() {
        let mut scheduler = ModuleLrScheduler::new(0.01_f64);
        for _ in 0..3 {
            check_approx(scheduler.step().base(), 0.01);
        }
    }

    #[test]
    fn step_advances_linear_default_scheduler() {
        let linear = LinearLrSchedulerConfig::new(0.9, 0.5, 4).build().unwrap();
        let mut scheduler = ModuleLrScheduler::new(linear);

        let expected = [0.9, 0.8, 0.7, 0.6, 0.5, 0.5];
        for expected_lr in expected {
            check_approx(scheduler.step().base(), expected_lr);
        }
    }

    #[test]
    fn save_load_preserves_default_scheduler_state() {
        let make =
            || ModuleLrScheduler::new(LinearLrSchedulerConfig::new(1.0, 0.1, 9).build().unwrap());

        let mut original = make();
        let mut truth = make();

        for _ in 0..5 {
            original.step();
            truth.step();
        }

        let record = original.to_record();
        let mut restored = make().load_record(record);

        for _ in 0..4 {
            check_approx(restored.step().base(), truth.step().base());
        }
    }

    #[test]
    fn group_param_gets_group_lr() {
        let id_group = ParamId::new();
        let id_default = ParamId::new();

        let mut scheduler = ModuleLrSchedulerConfig::new(0.001.into())
            .with_group(ParamGroup::from_ids(vec![id_group.clone()]), 0.1)
            .init()
            .unwrap();

        let policy = scheduler.step();
        // id_group is in the explicit group = group LR
        check_approx(policy.lr_from_param(id_group, None), 0.1);
        // id_default is not in any group = default LR
        check_approx(policy.lr_from_param(id_default, None), 0.001);
    }

    #[test]
    fn path_group_matches_param_by_path_substring() {
        let mut scheduler = ModuleLrSchedulerConfig::new(0.001.into())
            .with_group(ParamGroup::from_predicate("backbone"), 0.1)
            .init()
            .unwrap();

        let policy = scheduler.step();
        let id = ParamId::new();

        check_approx(
            policy.lr_from_param(id.clone(), Some("model.backbone.layer.weight")),
            0.1,
        );
        check_approx(
            policy.lr_from_param(id, Some("model.head.layer.weight")),
            0.001,
        );
    }

    #[test]
    fn multiple_groups_are_independent() {
        let id_a = ParamId::new();
        let id_b = ParamId::new();
        let id_default = ParamId::new();

        let mut scheduler =
            ModuleLrSchedulerConfig::new(LinearLrSchedulerConfig::new(0.001, 0.0001, 4).into())
                .with_group(
                    ParamGroup::from_ids(vec![id_a.clone()]),
                    LinearLrSchedulerConfig::new(0.1, 0.01, 4),
                )
                .with_group(
                    ParamGroup::from_ids(vec![id_b.clone()]),
                    LinearLrSchedulerConfig::new(0.5, 0.05, 4),
                )
                .init()
                .unwrap();

        // Each group returns its own initial LR
        let policy = scheduler.step();
        check_approx(policy.lr_from_param(id_a.clone(), None), 0.1);
        check_approx(policy.lr_from_param(id_b.clone(), None), 0.5);
        check_approx(policy.lr_from_param(id_default.clone(), None), 0.001);

        // All three schedulers advanced; LRs are strictly between initial and final
        let policy = scheduler.step();
        let lr_a = policy.lr_from_param(id_a, None);
        let lr_b = policy.lr_from_param(id_b, None);
        let lr_default = policy.lr_from_param(id_default, None);
        assert!(
            lr_a < 0.1 && lr_a > 0.01,
            "group-a LR should have decayed: {lr_a}"
        );
        assert!(
            lr_b < 0.5 && lr_b > 0.05,
            "group-b LR should have decayed: {lr_b}"
        );
        assert!(
            lr_default < 0.001 && lr_default > 0.0001,
            "default LR should have decayed: {lr_default}"
        );
    }

    #[test]
    fn save_load_with_groups_preserves_state() {
        let id_group = ParamId::new();

        let make = || {
            ModuleLrSchedulerConfig::new(LinearLrSchedulerConfig::new(0.01, 0.001, 9).into())
                .with_group(
                    ParamGroup::from_ids(vec![id_group.clone()]),
                    LinearLrSchedulerConfig::new(0.1, 0.01, 9),
                )
                .init()
                .unwrap()
        };

        let mut original = make();
        let mut truth = make();

        for _ in 0..5 {
            original.step();
            truth.step();
        }

        let record = original.to_record();
        let mut restored = make().load_record(record);

        for _ in 0..4 {
            let lr_restored = restored.step().lr_from_param(id_group.clone(), None);
            let lr_truth = truth.step().lr_from_param(id_group.clone(), None);
            check_approx(lr_restored, lr_truth);
        }
    }
}
