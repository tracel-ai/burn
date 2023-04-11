use super::state::{FormatOptions, NumericMetricState};
use super::{MetricEntry, MetricMetadata};
use crate::metric::{Metric, Numeric};
use burn_core::tensor::backend::Backend;
use burn_core::tensor::{ElementConversion, Int, Tensor};

/// The accuracy metric.
#[derive(Default)]
pub struct AccuracyMetric<B: Backend> {
    state: NumericMetricState,
    pad_token: Option<usize>,
    _b: B,
}

/// The [accuracy metric](AccuracyMetric) input type.
#[derive(new)]
pub struct AccuracyInput<B: Backend> {
    outputs: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
}

impl<B: Backend> AccuracyMetric<B> {
    /// Create the metric.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_pad_token(mut self, index: usize) -> Self {
        self.pad_token = Some(index);
        self
    }
}

impl<B: Backend> Metric for AccuracyMetric<B> {
    type Input = AccuracyInput<B>;

    fn update(&mut self, input: &AccuracyInput<B>, _metadata: &MetricMetadata) -> MetricEntry {
        let [batch_size, _n_classes] = input.outputs.dims();

        let targets = input.targets.clone().to_device(&B::Device::default());
        let outputs = input
            .outputs
            .clone()
            .argmax(1)
            .to_device(&B::Device::default())
            .reshape([batch_size]);

        let accuracy = match self.pad_token {
            Some(pad_token) => {
                let mask = targets.clone().equal_elem(pad_token as i64);
                let sames = outputs.equal(targets).into_int();
                let sames = sames.mask_fill(mask.clone(), 0);
                let num_pad = mask.into_int().sum().into_data().value[0].elem::<f64>();

                100.0
                    * (sames.sum().into_data().value[0].elem::<f64>()
                        / (batch_size as f64 - num_pad))
            }
            None => {
                let total_current =
                    Into::<i64>::into(outputs.equal(targets).into_int().sum().to_data().value[0])
                        as usize;
                100.0 * total_current as f64 / batch_size as f64
            }
        };

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
