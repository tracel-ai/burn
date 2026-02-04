use crate::renderer::{
    EvaluationProgress, MetricState, MetricsRenderer, MetricsRendererEvaluation,
    MetricsRendererTraining, ProgressType, TrainingProgress,
};

/// A simple renderer for when the cli feature is not enabled.
pub struct CliMetricsRenderer;

#[allow(clippy::new_without_default)]
impl CliMetricsRenderer {
    /// Create a new instance.
    pub fn new() -> Self {
        Self {}
    }
}

impl MetricsRendererTraining for CliMetricsRenderer {
    fn update_train(&mut self, _state: MetricState) {}

    fn update_valid(&mut self, _state: MetricState) {}

    fn render_train(
        &mut self,
        item: TrainingProgress,
        _progress_indicators: Vec<ProgressType>,
    ) {
        println!("{item:?}");
    }

    fn render_valid(
        &mut self,
        item: TrainingProgress,
        _progress_indicators: Vec<ProgressType>,
    ) {
        println!("{item:?}");
    }
}

impl MetricsRendererEvaluation for CliMetricsRenderer {
    fn render_test(
        &mut self,
        item: EvaluationProgress,
        _progress_indicators: Vec<ProgressType>,
    ) {
        println!("{item:?}");
    }

    fn update_test(&mut self, _name: super::EvaluationName, _state: MetricState) {}
}

impl MetricsRenderer for CliMetricsRenderer {
    fn manual_close(&mut self) {
        // Nothing to do.
    }

    fn register_metric(&mut self, _definition: crate::metric::MetricDefinition) {}
}
