use burn_core::tensor::{Bool, Device, FloatDType, Int, IntDType, Tensor};

use super::{
    ClassReduction, ConfusionStatsInput, Metric, MetricAttributes, MetricMetadata, MetricName,
    Numeric, NumericAttributes, NumericEntry, SerializedEntry,
    state::{ConfusionStatsState, FormatOptions},
};

/// Matthews correlation coefficient for binary and single-label multiclass classification.
///
/// Returns a coefficient in `[-1, 1]`, with `1` indicating perfect predictions.
/// A zero denominator (constant predictions or targets) produces `0`.
/// Epoch values are computed from accumulated counts, not averaged batch scores.
/// Multiclass MCC uses the generalized correlation, without macro/micro averaging.
/// Per-class integer counts are transferred to CPU for float64 accumulation and
/// computation, preserving accuracy for imbalanced classes without requiring GPU float64.
/// Updates require non-empty, finite predictions and a fixed number of classes
/// until [`clear`](Metric::clear) is called. Multi-label targets are not supported.
#[derive(Clone)]
pub struct MatthewsCorrelationCoefficientMetric {
    name: MetricName,
    threshold: Option<f64>,
    classes: Option<usize>,
    state: ConfusionStatsState,
}

impl Default for MatthewsCorrelationCoefficientMetric {
    fn default() -> Self {
        Self::binary(0.5)
    }
}

impl MatthewsCorrelationCoefficientMetric {
    /// Binary MCC for inputs of shape `[samples, 1]` containing positive-class
    /// probabilities and boolean targets. Probabilities strictly above `threshold`
    /// are positive. For two-column class scores, use [`Self::multiclass`].
    pub fn binary(threshold: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&threshold),
            "threshold must be between 0 and 1"
        );
        Self {
            name: format!("MatthewsCorrelationCoefficient @ Threshold({threshold})").into(),
            threshold: Some(threshold),
            classes: None,
            state: Default::default(),
        }
    }

    /// Multiclass MCC for `[samples, classes]` scores and one-hot boolean targets.
    /// The highest-scoring class is selected for each sample.
    /// Ties select the first class. Targets must contain exactly one true value per row.
    pub fn multiclass() -> Self {
        Self {
            name: "MatthewsCorrelationCoefficient @ Multiclass"
                .to_string()
                .into(),
            threshold: None,
            classes: None,
            state: Default::default(),
        }
    }

    fn coefficient(tp: Tensor<1>, fp: Tensor<1>, fn_: Tensor<1>) -> Tensor<1> {
        let targets = tp.clone() + fn_;
        let samples = targets.clone().sum();
        // Generalized MCC, evaluated in float64 like sklearn.metrics.matthews_corrcoef.
        let predicted = tp.clone() + fp;
        let covariance = tp.sum() * samples.clone() - (predicted.clone() * targets.clone()).sum();
        let samples_squared = samples.clone() * samples;
        let predicted_variance = samples_squared.clone() - (predicted.clone() * predicted).sum();
        let target_variance = samples_squared - (targets.clone() * targets).sum();
        let denominator = (predicted_variance * target_variance).sqrt();
        let undefined = denominator.clone().equal_scalar(0.0);
        (covariance / denominator)
            .mask_fill(undefined, 0.0)
            .clamp(-1.0, 1.0)
    }
}

impl Metric for MatthewsCorrelationCoefficientMetric {
    type Input = ConfusionStatsInput;

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        assert_eq!(input.predictions.dims(), input.targets.dims());
        let [sample_size, classes] = input.predictions.dims();
        assert!(
            sample_size > 0 && classes > 0,
            "MCC requires non-empty input"
        );
        if let Some(previous) = self.classes {
            assert_eq!(
                classes, previous,
                "MCC class count must remain constant between batches"
            );
        }
        assert!(
            input
                .predictions
                .clone()
                .is_finite()
                .all()
                .into_scalar::<bool>(),
            "MCC predictions must be finite"
        );
        let (predicted, targets) = match self.threshold {
            Some(threshold) => {
                assert_eq!(classes, 1, "binary MCC expects one positive-class column");
                let positive = input.predictions.clone().greater_scalar(threshold);
                // Include the negative class so the same generalized formula
                // covers binary MCC, including its true-negative count.
                (
                    Tensor::cat(vec![positive.clone().bool_not(), positive], 1),
                    Tensor::cat(
                        vec![input.targets.clone().bool_not(), input.targets.clone()],
                        1,
                    ),
                )
            }
            None => {
                assert!(
                    input
                        .targets
                        .clone()
                        .cast(IntDType::I64)
                        .sum_dim(1)
                        .equal_scalar(1)
                        .all()
                        .into_scalar::<bool>(),
                    "multiclass MCC requires one-hot targets"
                );
                let indices =
                    Tensor::<1, Int>::arange(0..classes as i64, &input.predictions.device())
                        .unsqueeze_dim(0);
                (
                    input.predictions.clone().argmax(1).equal(indices),
                    input.targets.clone(),
                )
            }
        };
        let count = |mask: Tensor<2, Bool>| {
            mask.cast(IntDType::I64)
                .sum_dim(0)
                .squeeze_dim(0)
                .to_device(&Device::flex())
                .cast(FloatDType::F64)
        };
        self.classes = Some(classes);
        self.state.update(
            Some(count(predicted.clone().bool_and(targets.clone()))),
            Some(count(
                predicted.clone().bool_and(targets.clone().bool_not()),
            )),
            Some(count(predicted.bool_not().bool_and(targets))),
            sample_size,
        );
        self.state.compute_update(
            ClassReduction::Micro,
            FormatOptions::new(self.name()).precision(4),
            |tp, fp, fn_| Self::coefficient(tp.unwrap(), fp.unwrap(), fn_.unwrap()),
        )
    }

    fn compute(&mut self) -> SerializedEntry {
        self.state
            .compute_final(FormatOptions::new(self.name()).precision(4))
    }

    fn clear(&mut self) {
        self.state.reset();
        self.classes = None;
    }

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: None,
            higher_is_better: true,
        }
        .into()
    }
}

impl Numeric for MatthewsCorrelationCoefficientMetric {
    fn value(&self) -> Option<NumericEntry> {
        self.state.current_value()
    }

    fn running_value(&self) -> Option<NumericEntry> {
        self.state.running_value()
    }

    fn final_value(&self) -> NumericEntry {
        self.state.final_value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_core::tensor::DType;
    use rstest::rstest;

    fn binary_input(predictions: [[f32; 1]; 4], targets: [[i32; 1]; 4]) -> ConfusionStatsInput {
        let device = Default::default();
        ConfusionStatsInput::new(
            Tensor::from_data(predictions, &device),
            Tensor::from_data(targets, &device),
        )
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
    }

    #[rstest]
    #[case::perfect([[0.1], [0.9], [0.1], [0.9]], 1.0)]
    #[case::inverse([[0.9], [0.1], [0.9], [0.1]], -1.0)]
    #[case::uncorrelated([[0.1], [0.1], [0.9], [0.9]], 0.0)]
    #[case::constant_predictions([[0.9]; 4], 0.0)]
    #[case::threshold_tie([[0.5], [0.9], [0.5], [0.9]], 1.0)]
    fn binary(#[case] predictions: [[f32; 1]; 4], #[case] expected: f64) {
        let input = binary_input(predictions, [[0], [1], [0], [1]]);
        let mut metric = MatthewsCorrelationCoefficientMetric::default();
        metric.update(&input, &MetricMetadata::fake());
        assert_close(metric.value().unwrap().current(), expected);
        metric.compute();
        assert_close(metric.final_value().current(), expected);
    }

    #[rstest]
    #[case::constant_targets([[0], [0], [0], [0]], 0.0)]
    // TP=2, TN=0, FP=1, FN=1: MCC = -1/3.
    #[case::imbalanced([[1], [1], [1], [0]], -1.0 / 3.0)]
    fn binary_targets(#[case] targets: [[i32; 1]; 4], #[case] expected: f64) {
        let input = binary_input([[0.9], [0.1], [0.9], [0.9]], targets);
        let mut metric = MatthewsCorrelationCoefficientMetric::default();
        metric.update(&input, &MetricMetadata::fake());
        assert_close(metric.value().unwrap().current(), expected);
    }

    #[test]
    fn custom_threshold() {
        let input = binary_input([[0.6], [0.9], [0.6], [0.9]], [[0], [1], [0], [1]]);
        let mut metric = MatthewsCorrelationCoefficientMetric::binary(0.75);
        metric.update(&input, &MetricMetadata::fake());
        assert_close(metric.value().unwrap().current(), 1.0);
        assert_ne!(
            metric.name(),
            MatthewsCorrelationCoefficientMetric::default().name()
        );
    }

    #[test]
    fn multiclass_and_batch_accumulation() {
        let device = Default::default();
        // Confusion matrix [[2, 1, 0], [0, 1, 1], [1, 0, 0]].
        // c=3, s=6, predicted=[3,2,1], targets=[3,2,1]: MCC=(18-14)/(36-14).
        let input = ConfusionStatsInput::new(
            Tensor::<1, Int>::from_data([0, 0, 1, 1, 2, 0], &device)
                .one_hot(3)
                .float(),
            Tensor::<1, Int>::from_data([0, 0, 0, 1, 1, 2], &device)
                .one_hot(3)
                .bool(),
        );
        let mut whole = MatthewsCorrelationCoefficientMetric::multiclass();
        whole.update(&input, &MetricMetadata::fake());
        assert_close(whole.value().unwrap().current(), 2.0 / 11.0);

        let mut split = MatthewsCorrelationCoefficientMetric::multiclass();
        for range in [0..2, 2..6] {
            split.update(
                &ConfusionStatsInput::new(
                    input.predictions.clone().slice([range.clone()]),
                    input.targets.clone().slice([range]),
                ),
                &MetricMetadata::fake(),
            );
        }
        assert_close(split.running_value().unwrap().current(), 2.0 / 11.0);
        split.compute();
        assert_close(split.final_value().current(), whole.final_value().current());
        split.clear();
        assert!(split.value().is_none());
        assert!(split.running_value().is_none());
        split.update(&input, &MetricMetadata::fake());
        assert_close(split.final_value().current(), 2.0 / 11.0);
    }

    #[test]
    fn binary_accumulates_counts() {
        let mut metric = MatthewsCorrelationCoefficientMetric::default();
        // Each constant-class batch has MCC=0, but together they are perfect.
        for value in [0, 1] {
            let input = binary_input([[value as f32]; 4], [[value]; 4]);
            metric.update(&input, &MetricMetadata::fake());
            assert_close(metric.value().unwrap().current(), 0.0);
        }
        assert_close(metric.running_value().unwrap().current(), 1.0);
        let entry = metric.compute();
        assert!(matches!(
            NumericEntry::deserialize(&entry.serialized).unwrap(),
            NumericEntry::Final(value) if (value - 1.0).abs() < 1e-6
        ));
        assert_close(metric.final_value().current(), 1.0);
    }

    #[rstest]
    #[case::perfect([0, 1, 2], [0, 1, 2], 1.0)]
    #[case::cyclic_errors([1, 2, 0], [0, 1, 2], -0.5)]
    #[case::constant_predictions([0, 0, 0], [0, 1, 2], 0.0)]
    #[case::constant_targets([0, 1, 2], [0, 0, 0], 0.0)]
    #[case::absent_class([0, 1, 0], [0, 1, 0], 1.0)]
    fn multiclass(#[case] predictions: [i32; 3], #[case] targets: [i32; 3], #[case] expected: f64) {
        let device = Default::default();
        let input = ConfusionStatsInput::new(
            Tensor::<1, Int>::from_data(predictions, &device)
                .one_hot(3)
                .float(),
            Tensor::<1, Int>::from_data(targets, &device)
                .one_hot(3)
                .bool(),
        );
        let mut metric = MatthewsCorrelationCoefficientMetric::multiclass();
        metric.update(&input, &MetricMetadata::fake());
        assert_close(metric.value().unwrap().current(), expected);
    }

    // References generated with scikit-learn 1.9.0:
    // t, p = np.indices(C.shape)
    // matthews_corrcoef(t.ravel(), p.ravel(), sample_weight=C.ravel())
    // Integer weights represent repeated observations without huge test tensors.
    #[rstest]
    #[case::asymmetric([[30, 2, 1], [5, 9, 2], [0, 4, 7]], 0.6000253345357297)]
    #[case::unseen_target_class([[9, 0, 0], [0, 0, 0], [1, 2, 0]], 0.3872983346207417)]
    #[case::all_wrong([[0, 2, 3], [4, 0, 5], [6, 7, 0]], -0.501322567268291)]
    #[case::large_perfect([[16_777_216, 0, 0], [0, 1, 0], [0, 0, 0]], 1.0)]
    #[case::large_one_error([[16_777_216, 1, 0], [0, 1, 0], [0, 0, 0]], 0.7071067601131242)]
    #[case::billion_samples([[999_999_999, 1, 0], [1, 0, 0], [0, 0, 1]], 0.5)]
    fn sklearn_count_reference(#[case] matrix: [[u64; 3]; 3], #[case] expected: f64) {
        let device = Device::flex();
        let tp = std::array::from_fn::<_, 3, _>(|k| matrix[k][k] as f64);
        let fp = std::array::from_fn::<_, 3, _>(|k| {
            matrix.iter().map(|row| row[k]).sum::<u64>() as f64 - tp[k]
        });
        let fn_ = std::array::from_fn::<_, 3, _>(|k| matrix[k].iter().sum::<u64>() as f64 - tp[k]);
        let actual = MatthewsCorrelationCoefficientMetric::coefficient(
            Tensor::from_data(tp, (&device, DType::F64)),
            Tensor::from_data(fp, (&device, DType::F64)),
            Tensor::from_data(fn_, (&device, DType::F64)),
        )
        .into_scalar::<f64>();
        assert_close(actual, expected);
    }

    #[rstest]
    #[case::binary(FloatDType::F32, true)]
    #[case::multiclass(FloatDType::F32, false)]
    #[case::half_precision(FloatDType::F16, false)]
    fn sklearn_imbalanced_reference(#[case] dtype: FloatDType, #[case] binary: bool) {
        use burn_core::tensor::TensorData;
        let device = Device::flex();
        let mut labels = vec![0i64; 10_000];
        labels[9_999] = 1;
        let targets =
            Tensor::<1, Int>::from_data(TensorData::new(labels.clone(), [10_000]), &device);
        labels[9_998] = 1;
        let predicted = Tensor::<1, Int>::from_data(TensorData::new(labels, [10_000]), &device);
        let (input, mut metric) = if binary {
            (
                ConfusionStatsInput::new(
                    predicted.float().reshape([10_000, 1]),
                    targets.bool().reshape([10_000, 1]),
                ),
                MatthewsCorrelationCoefficientMetric::default(),
            )
        } else {
            (
                ConfusionStatsInput::new(predicted.one_hot(3).float(), targets.one_hot(3).bool()),
                MatthewsCorrelationCoefficientMetric::multiclass(),
            )
        };
        let input = ConfusionStatsInput::new(input.predictions.cast(dtype), input.targets);
        metric.update(&input, &MetricMetadata::fake());
        // sklearn.metrics.matthews_corrcoef([0]*9999+[1], [0]*9998+[1,1])
        assert_close(metric.final_value().current(), 0.7070714214274962);
    }

    #[test]
    fn score_ties_choose_first_class() {
        let device = Device::flex();
        let input = ConfusionStatsInput::new(
            Tensor::from_data([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]], &device),
            Tensor::from_data([[1, 0, 0], [0, 1, 0], [0, 0, 1]], &device),
        );
        let mut metric = MatthewsCorrelationCoefficientMetric::multiclass();
        metric.update(&input, &MetricMetadata::fake());
        assert_close(metric.final_value().current(), 1.0);
    }

    #[test]
    fn single_class_is_zero() {
        let device = Device::flex();
        let input = ConfusionStatsInput::new(
            Tensor::from_data([[0.7]], &device),
            Tensor::from_data([[true]], &device),
        );
        let mut metric = MatthewsCorrelationCoefficientMetric::multiclass();
        metric.update(&input, &MetricMetadata::fake());
        assert_close(metric.final_value().current(), 0.0);
    }

    #[rstest]
    #[case::nan(f32::NAN)]
    #[case::positive_infinity(f32::INFINITY)]
    #[case::negative_infinity(f32::NEG_INFINITY)]
    #[should_panic(expected = "MCC predictions must be finite")]
    fn rejects_nonfinite_predictions(#[case] value: f32) {
        let input = binary_input([[value]; 4], [[0], [1], [0], [1]]);
        MatthewsCorrelationCoefficientMetric::default().update(&input, &MetricMetadata::fake());
    }

    #[rstest]
    #[case::multi_hot([[1, 1], [0, 1]])]
    #[case::missing_label([[0, 0], [0, 1]])]
    #[should_panic(expected = "multiclass MCC requires one-hot targets")]
    fn rejects_invalid_targets(#[case] targets: [[i32; 2]; 2]) {
        let device = Device::flex();
        let input = ConfusionStatsInput::new(
            Tensor::from_data([[0.8, 0.2], [0.1, 0.9]], &device),
            Tensor::from_data(targets, &device),
        );
        MatthewsCorrelationCoefficientMetric::multiclass().update(&input, &MetricMetadata::fake());
    }

    #[test]
    #[should_panic(expected = "MCC requires non-empty input")]
    fn rejects_empty_input() {
        let device = Device::flex();
        let input = ConfusionStatsInput::new(
            Tensor::zeros([0, 1], &device),
            Tensor::zeros([0, 1], &device),
        );
        MatthewsCorrelationCoefficientMetric::default().update(&input, &MetricMetadata::fake());
    }

    #[test]
    #[should_panic(expected = "MCC class count must remain constant between batches")]
    fn rejects_changing_classes() {
        let device = Device::flex();
        let mut metric = MatthewsCorrelationCoefficientMetric::multiclass();
        for classes in [2, 3] {
            let labels = Tensor::<1, Int>::from_data([0, 1], &device).one_hot(classes);
            let input = ConfusionStatsInput::new(labels.clone().float(), labels.bool());
            metric.update(&input, &MetricMetadata::fake());
        }
    }
}
