use crate::{
    components::LearnerComponents, metric::Metric, Aggregate, Direction, EventCollector, Split,
};

/// Break of the training loop when some conditions are meet.
pub enum EarlyStopping {
    /// Strategy based on metrics.
    Metric(MetricEarlyStoppingStrategy),
}

impl EarlyStopping {
    pub(crate) fn should_stop<LC: LearnerComponents>(
        &mut self,
        epoch: usize,
        info: &mut LC::EventCollector,
    ) -> bool {
        match self {
            EarlyStopping::Metric(strategy) => strategy.should_stop::<LC>(epoch, info),
        }
    }
}

/// The condition that [early stopping strategies](EarlyStoppingStrategy) should follow.
pub enum StoppingCondition {
    /// When no improvement has happended since the given number of epochs.
    /// In other words, when no best epoch has been found.
    NoImprovementSince {
        /// The number of epochs allowed to get worsen before it get better.
        n_epochs: usize,
    },
}

/// A strategy that checks if the training should be stopped.
pub trait EarlyStoppingStrategy: Into<EarlyStopping> {
    /// Update its current state and returns if the training should be stopped.
    fn should_stop<LC: LearnerComponents>(
        &mut self,
        epoch: usize,
        info: &mut LC::EventCollector,
    ) -> bool;
}

/// An [early stopping strategy](EarlyStoppingStrategy) based on a metrics collected
/// during training or validation.
pub struct MetricEarlyStoppingStrategy {
    condition: StoppingCondition,
    metric_name: String,
    aggregate: Aggregate,
    direction: Direction,
    split: Split,
    best_epoch: usize,
    best_value: f64,
}

impl EarlyStoppingStrategy for MetricEarlyStoppingStrategy {
    fn should_stop<LC: LearnerComponents>(
        &mut self,
        epoch: usize,
        info: &mut LC::EventCollector,
    ) -> bool {
        let current_value =
            match info.find_metric(&self.metric_name, epoch, self.aggregate, self.split) {
                Some(value) => value,
                None => {
                    log::warn!("Can't find metric for early stopping.");
                    return false;
                }
            };

        let is_best = match self.direction {
            Direction::Lowest => current_value <= self.best_value,
            Direction::Highest => current_value >= self.best_value,
        };

        if is_best {
            log::info!(
                "New best epoch found {} {}: {}",
                epoch,
                self.metric_name,
                current_value
            );
            self.best_value = current_value;
            self.best_epoch = epoch;
            return false;
        }

        match self.condition {
            StoppingCondition::NoImprovementSince { n_epochs } => {
                let should_stop = epoch - self.best_epoch > n_epochs;

                if should_stop {
                    log::info!("Stopping training loop, no improvement since epoch {}, {}: {},  current epoch {}, {}: {}", self.best_epoch, self.metric_name, self.best_value, epoch, self.metric_name, current_value);
                }

                should_stop
            }
        }
    }
}

impl Into<EarlyStopping> for MetricEarlyStoppingStrategy {
    fn into(self) -> EarlyStopping {
        EarlyStopping::Metric(self)
    }
}

impl MetricEarlyStoppingStrategy {
    /// Create a new [early stopping strategy](EarlyStoppingStrategy) based on a metrics collected
    /// during training or validation.
    ///
    /// # Notes
    ///
    /// The metric should be registered for early stopping to work, otherwise no data is collected.
    pub fn new<Me: Metric>(
        aggregate: Aggregate,
        direction: Direction,
        split: Split,
        condition: StoppingCondition,
    ) -> Self {
        let init_value = match direction {
            Direction::Lowest => f64::MAX,
            Direction::Highest => f64::MIN,
        };

        Self {
            metric_name: Me::NAME.to_string(),
            condition,
            aggregate,
            direction,
            split,
            best_epoch: 1,
            best_value: init_value,
        }
    }
}
