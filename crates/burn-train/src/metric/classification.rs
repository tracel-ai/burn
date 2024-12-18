use std::num::NonZeroUsize;

/// Necessary data for classification metrics.
#[derive(Default)]
pub struct ClassificationMetricConfig {
    pub decision_rule: DecisionRule,
    pub class_reduction: ClassReduction,
}

/// The prediction decision rule for classification metrics.
pub enum DecisionRule {
    Threshold(f64),
    TopK(NonZeroUsize),
}

impl Default for DecisionRule {
    fn default() -> Self {
        Self::Threshold(0.5)
    }
}

/// The reduction strategy for classification metrics.
#[derive(Copy, Clone, Default)]
pub enum ClassReduction {
    /// Computes the statistics over all classes before averaging
    Micro,
    /// Computes the statistics independently for each class before averaging
    #[default]
    Macro,
}
