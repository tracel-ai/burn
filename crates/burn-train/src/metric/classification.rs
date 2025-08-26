use std::num::NonZeroUsize;

/// Necessary data for classification metrics.
#[derive(Default, Debug, Clone)]
pub struct ClassificationMetricConfig {
    pub decision_rule: DecisionRule,
    pub class_reduction: ClassReduction,
}

/// The prediction decision rule for classification metrics.
#[derive(Debug, Clone)]
pub enum DecisionRule {
    /// Consider a class predicted if its probability exceeds the threshold.
    Threshold(f64),
    /// Consider a class predicted correctly if it is within the top k predicted classes based on scores.
    TopK(NonZeroUsize),
}

impl Default for DecisionRule {
    fn default() -> Self {
        Self::Threshold(0.5)
    }
}

/// The reduction strategy for classification metrics.
#[derive(Copy, Clone, Default, Debug)]
pub enum ClassReduction {
    /// Computes the statistics over all classes before averaging
    Micro,
    /// Computes the statistics independently for each class before averaging
    #[default]
    Macro,
}
