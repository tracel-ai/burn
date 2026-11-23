use burn_optim::lr_scheduler::module_lr_scheduler::ModuleLearningRate;

/// Learning rates used for one applied optimizer update.
pub struct OptimizerStepOutput {
    learning_rate: ModuleLearningRate,
}

impl OptimizerStepOutput {
    /// Capture the learning rates used for this update.
    pub fn new(learning_rate: ModuleLearningRate) -> Self {
        Self { learning_rate }
    }

    /// Learning rates used for this update, in scheduler registration order.
    pub fn learning_rate(&self) -> &ModuleLearningRate {
        &self.learning_rate
    }
}
