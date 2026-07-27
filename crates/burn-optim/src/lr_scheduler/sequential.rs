use alloc::vec::Vec;
use burn_core as burn;

use burn::config::Config;

use super::module_lr_scheduler::ModuleLrScheduler;
use super::{DynLrScheduler, LrScheduler, LrSchedulerConfig, LrSchedulerRecord, String};
use crate::{LearningRate, RecordState};

/// Configuration for a [sequential learning rate scheduler](SequentialLrScheduler).
///
/// Each milestone is the number of completed calls to [`LrScheduler::step`] at which the next
/// scheduler takes over. Therefore, `N` schedulers require exactly `N - 1` milestones. Milestones
/// must be greater than zero and strictly increasing.
///
/// # Example
///
/// The linear scheduler below handles the first 100 steps, after which the cosine scheduler starts
/// from its own first step.
///
/// ```
/// use burn_optim::lr_scheduler::{
///     LrSchedulerConfig,
///     cosine::CosineAnnealingLrSchedulerConfig, linear::LinearLrSchedulerConfig,
///     sequential::SequentialLrSchedulerConfig,
/// };
///
/// let config = SequentialLrSchedulerConfig::new(
///     vec![
///         LrSchedulerConfig::Linear(LinearLrSchedulerConfig::new(1e-5, 1e-2, 100)),
///         LrSchedulerConfig::Cosine(CosineAnnealingLrSchedulerConfig::new(1e-2, 900)),
///     ],
///     vec![100],
/// );
/// let scheduler = config.init().unwrap();
/// ```
#[derive(Config, Debug)]
pub struct SequentialLrSchedulerConfig {
    schedulers: Vec<LrSchedulerConfig>,
    milestones: Vec<usize>,
}

impl SequentialLrSchedulerConfig {
    pub(crate) fn build(&self) -> Result<SequentialLrScheduler, String> {
        if self.schedulers.is_empty() {
            return Err("At least one scheduler is required".into());
        }
        if self.milestones.len() + 1 != self.schedulers.len() {
            return Err(
                "The number of milestones must be one less than the number of schedulers".into(),
            );
        }
        if self
            .milestones
            .iter()
            .enumerate()
            .any(|(index, milestone)| {
                *milestone == 0 || index > 0 && *milestone <= self.milestones[index - 1]
            })
        {
            return Err("Milestones must be greater than zero and strictly increasing".into());
        }

        let schedulers = self
            .schedulers
            .iter()
            .map(LrSchedulerConfig::build)
            .collect::<Result<Vec<_>, _>>()?;

        Ok(SequentialLrScheduler {
            schedulers,
            milestones: self.milestones.clone(),
            step: 0,
        })
    }

    /// Initializes a [module learning rate scheduler](ModuleLrScheduler).
    ///
    /// # Errors
    ///
    /// An error is returned when there are no schedulers, the number of milestones is not one less
    /// than the number of schedulers, milestones are zero or not strictly increasing, or a child
    /// scheduler configuration is invalid.
    pub fn init(&self) -> Result<ModuleLrScheduler, String> {
        self.build().map(Into::into)
    }
}

/// Runs learning rate schedulers one after another at configured milestones.
#[derive(Clone)]
pub struct SequentialLrScheduler {
    schedulers: Vec<DynLrScheduler>,
    milestones: Vec<usize>,
    step: usize,
}

impl SequentialLrScheduler {
    fn active_scheduler(&self) -> usize {
        self.milestones.partition_point(|&m| self.step >= m)
    }
}

impl LrScheduler for SequentialLrScheduler {
    fn step(&mut self) -> LearningRate {
        let index = self.active_scheduler();
        let lr = self.schedulers[index].step();
        self.step = self
            .step
            .checked_add(1)
            .expect("The sequential scheduler step counter overflowed");
        lr
    }

    fn to_record(&self) -> LrSchedulerRecord {
        let mut record =
            LrSchedulerRecord::from_state(&SequentialLrSchedulerState { step: self.step });
        for (index, scheduler) in self.schedulers.iter().enumerate() {
            record = record.with_record(&index.to_string(), scheduler.to_record());
        }
        record
    }

    fn load_record(&mut self, record: LrSchedulerRecord) {
        if let Some(state) = record.into_state::<SequentialLrSchedulerState>() {
            self.step = state.step;
        }

        let schedulers = core::mem::take(&mut self.schedulers);
        self.schedulers = schedulers
            .into_iter()
            .enumerate()
            .map(|(index, scheduler)| scheduler.load_record(record.record(&index.to_string())))
            .collect();
    }
}

#[derive(RecordState, Clone, Debug)]
struct SequentialLrSchedulerState {
    step: usize,
}

#[cfg(test)]
mod tests {
    use super::super::cosine::CosineAnnealingLrSchedulerConfig;
    use super::super::exponential::ExponentialLrSchedulerConfig;
    use super::super::linear::LinearLrSchedulerConfig;
    use super::super::test_utils;
    use super::*;

    fn config(milestones: Vec<usize>) -> SequentialLrSchedulerConfig {
        SequentialLrSchedulerConfig::new(
            vec![
                LinearLrSchedulerConfig::new(0.1, 0.3, 2).into(),
                ExponentialLrSchedulerConfig::new(0.5, 0.5).into(),
                CosineAnnealingLrSchedulerConfig::new(0.8, 2).into(),
            ],
            milestones,
        )
    }

    #[test]
    fn switches_schedulers_at_milestones() {
        let scheduler = config(vec![2, 5]).build().unwrap();
        test_utils::check_lr_sequence(scheduler, [0.1, 0.2, 0.5, 0.25, 0.125, 0.8, 0.4, 0.0]);
    }

    #[test]
    fn rejects_empty_scheduler_list() {
        let result = SequentialLrSchedulerConfig::new(vec![], vec![]).build();
        assert_eq!(result.err().unwrap(), "At least one scheduler is required");
    }

    #[test]
    fn rejects_wrong_milestone_count() {
        let result = config(vec![2]).build();
        assert_eq!(
            result.err().unwrap(),
            "The number of milestones must be one less than the number of schedulers"
        );
    }

    #[test]
    fn rejects_zero_or_non_increasing_milestones() {
        for milestones in [vec![0, 2], vec![2, 2], vec![3, 2]] {
            let result = config(milestones).build();
            assert_eq!(
                result.err().unwrap(),
                "Milestones must be greater than zero and strictly increasing"
            );
        }
    }

    #[test]
    fn reports_invalid_child_config() {
        let result = SequentialLrSchedulerConfig::new(
            vec![LinearLrSchedulerConfig::new(0.1, 0.2, 0).into()],
            vec![],
        )
        .build();
        assert_eq!(
            result.err().unwrap(),
            "Number of iterations must be at least 1"
        );
    }

    #[test]
    fn saves_and_loads_before_and_after_transitions() {
        test_utils::check_save_load(config(vec![2, 5]).build().unwrap(), 1);
        test_utils::check_save_load(config(vec![2, 5]).build().unwrap(), 4);
        test_utils::check_save_load(config(vec![2, 5]).build().unwrap(), 6);
    }
}
