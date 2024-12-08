use std::num::NonZeroUsize;

/// The reduction strategy for classification metrics.
#[derive(Copy, Clone, Default)]
pub enum ClassReduction {
    /// Computes the statistics over all classes before averaging
    Micro,
    /// Computes the statistics independently for each class before averaging
    #[default]
    Macro,
}

#[derive(Default)]
pub struct ClassificationMetricConfig {
    pub decision_rule: ClassificationDecisionRule,
    pub class_reduction: ClassReduction,
}

pub enum ClassificationDecisionRule {
    Threshold(f64),
    TopK(NonZeroUsize),
}

impl Default for ClassificationDecisionRule {
    fn default() -> Self {
        Self::Threshold(0.5)
    }
}
