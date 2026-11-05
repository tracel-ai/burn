use super::MetricMetadata;
use super::state::{FormatOptions, NumericMetricState};
use crate::metric::{
    Metric, MetricAttributes, MetricName, Numeric, NumericAttributes, NumericEntry,
    RegressionInput, SerializedEntry,
};
use burn_core::tensor::TensorReadError;

/// The Mean Squared Error (MSE) over all prediction values, including all target columns.
///
/// # Panics
///
/// Updating the metric panics if outputs and targets have different shapes or are empty.
#[derive(Clone)]
pub struct MseMetric {
    name: MetricName,
    state: NumericMetricState,
}

impl Default for MseMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl MseMetric {
    /// Creates the metric.
    pub fn new() -> Self {
        Self {
            name: MetricName::new("MSE".to_string()),
            state: NumericMetricState::default(),
        }
    }
}

impl Metric for MseMetric {
    type Input = RegressionInput;

    fn update(
        &mut self,
        input: &Self::Input,
        _metadata: &MetricMetadata,
    ) -> Result<SerializedEntry, TensorReadError> {
        let targets = input.targets.clone();
        let outputs = input.outputs.clone();

        assert_eq!(outputs.dims(), targets.dims(), "MSE shapes must match");
        let [batch_size, num_targets] = outputs.dims();
        assert!(
            batch_size > 0 && num_targets > 0,
            "MSE requires non-empty input"
        );

        let diff = outputs.sub(targets);
        let mse = diff.powf_scalar(2.0).mean();

        self.state
            .update(mse.try_into_scalar::<f64>()?, batch_size * num_targets);
        Ok(self
            .state
            .compute_update(FormatOptions::new(self.name()).precision(4)))
    }

    fn compute(&mut self) -> Result<SerializedEntry, TensorReadError> {
        Ok(self
            .state
            .compute_final(FormatOptions::new(self.name()).precision(4)))
    }

    fn clear(&mut self) {
        self.state.reset()
    }

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: None,
            higher_is_better: false,
        }
        .into()
    }
}

impl Numeric for MseMetric {
    fn value(&self) -> Option<NumericEntry> {
        Some(self.state.current_value())
    }

    fn running_value(&self) -> Option<NumericEntry> {
        Some(self.state.running_value())
    }

    fn final_value(&self) -> NumericEntry {
        self.state.final_value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_core::tensor::Tensor;

    #[test]
    #[should_panic(expected = "MSE shapes must match")]
    fn rejects_mismatched_shapes() {
        let device = Default::default();
        MseMetric::new()
            .update(
                &RegressionInput::new(
                    Tensor::from_data([[1., 2.]], &device),
                    Tensor::from_data([[1.]], &device),
                ),
                &MetricMetadata::fake(),
            )
            .unwrap();
    }

    #[test]
    #[should_panic(expected = "MSE requires non-empty input")]
    fn rejects_empty_input() {
        let device = Default::default();
        MseMetric::new()
            .update(
                &RegressionInput::new(
                    Tensor::empty([1, 0], &device),
                    Tensor::empty([1, 0], &device),
                ),
                &MetricMetadata::fake(),
            )
            .unwrap();
    }

    #[test]
    fn weights_all_target_values_across_batches() {
        let device = Default::default();
        let mut metric = MseMetric::new();
        metric
            .update(
                &RegressionInput::new(
                    Tensor::from_data([[0.]], &device),
                    Tensor::from_data([[0.]], &device),
                ),
                &MetricMetadata::fake(),
            )
            .unwrap();
        metric
            .update(
                &RegressionInput::new(
                    Tensor::from_data([[0., 0., 0.]], &device),
                    Tensor::from_data([[2., 2., 2.]], &device),
                ),
                &MetricMetadata::fake(),
            )
            .unwrap();
        assert_eq!(metric.final_value().current(), 3.0);
    }

    #[test]
    fn sklearn_reference_across_batches_and_epochs() {
        // scikit-learn 1.8.0: mean_squared_error(y_true, y_pred) == 1.025.
        let device = Default::default();
        let targets = Tensor::from_data(
            [[1., 10.], [2., 10.], [3., 12.], [4., 14.], [5., 18.]],
            &device,
        );
        let outputs = Tensor::from_data(
            [[1., 9.], [2.5, 11.], [2., 11.], [5., 14.], [4., 20.]],
            &device,
        );
        let mut metric = MseMetric::default();
        assert_eq!(metric.name().as_str(), "MSE");
        assert!(matches!(
            metric.attributes(),
            MetricAttributes::Numeric(NumericAttributes {
                unit: None,
                higher_is_better: false,
            })
        ));

        for range in [0..2, 2..5] {
            metric
                .update(
                    &RegressionInput::new(
                        outputs.clone().slice([range.clone()]),
                        targets.clone().slice([range]),
                    ),
                    &MetricMetadata::fake(),
                )
                .unwrap();
        }
        assert!(matches!(
            metric.value().unwrap(),
            NumericEntry::Aggregated { aggregated_value, count: 6 }
                if (aggregated_value - (4.0 / 3.0)).abs() < 1e-6
        ));
        assert!((metric.running_value().unwrap().current() - 1.025).abs() < 1e-6);
        let entry = metric.compute().unwrap();
        assert!(matches!(
            NumericEntry::deserialize(&entry.serialized).unwrap(),
            NumericEntry::Final(value) if (value - 1.025).abs() < 1e-6
        ));
        assert!((metric.final_value().current() - 1.025).abs() < 1e-6);

        metric.clear();
        metric
            .update(
                &RegressionInput::new(outputs.slice(0..2), targets.slice(0..2)),
                &MetricMetadata::fake(),
            )
            .unwrap();
        metric.compute().unwrap();
        assert!((metric.final_value().current() - 0.5625).abs() < 1e-6);
    }
}
