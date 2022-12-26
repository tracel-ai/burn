use super::state::{FormatOptions, NumericMetricState};
use super::MetricEntry;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;
use crate::train::metric::{Metric, Numeric};

/// The accuracy metric.
#[derive(Default)]
pub struct AccuracyMetric<B: Backend> {
    state: NumericMetricState,
    _b: B,
}

/// The [accuracy metric](AccuracyMetric) input type.
#[derive(new)]
pub struct AccuracyInput<B: Backend> {
    outputs: Tensor<B, 2>,
    targets: Tensor<B::IntegerBackend, 1>,
}

impl<B: Backend> AccuracyMetric<B> {
    /// Create the metric.
    pub fn new() -> Self {
        Self::default()
    }
}

impl<B: Backend> Metric for AccuracyMetric<B> {
    type Input = AccuracyInput<B>;

    fn update(&mut self, input: &AccuracyInput<B>) -> MetricEntry {
        let [batch_size, _n_classes] = input.outputs.dims();

        let targets = input.targets.to_device(B::Device::default());
        let outputs = input
            .outputs
            .argmax(1)
            .to_device(B::Device::default())
            .reshape([batch_size]);

        let total_current = outputs.equal(&targets).to_int().sum().to_data().value[0] as usize;
        let accuracy = 100.0 * total_current as f64 / batch_size as f64;

        self.state.update(
            accuracy,
            batch_size,
            FormatOptions::new("Accuracy").unit("%").precision(2),
        )
    }

    fn clear(&mut self) {
        self.state.reset()
    }
}

impl<B: Backend> Numeric for AccuracyMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}
