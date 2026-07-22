use std::sync::Arc;

use burn_core::tensor::{Bool, Tensor};

use crate::metric::{ClassReduction, MetricName, NumericEntry, SerializedEntry, format_float};

/// Useful utility to implement numeric metrics.
///
/// # Notes
///
/// The numeric metric store values inside floats.
/// Even if some metric are integers, their mean are floats.
#[derive(Clone)]
pub struct NumericMetricState {
    sum: f64,
    count: usize,
    current: f64,
    current_count: usize,
}

/// Accumulates raw predictions and targets across batches.
///
/// Used by rank-based metrics (AUROC, AUC-PR) that must recompute over the
/// whole epoch. Buffers are freed on [`reset`](Self::reset).
#[derive(Clone)]
pub struct PredictionAccumulatorState {
    predictions: Vec<Tensor<2>>,
    targets: Vec<Tensor<2, Bool>>,
    current: Option<f64>,
}

/// Formatting options for the [numeric metric state](NumericMetricState).
pub struct FormatOptions {
    name: Arc<String>,
    unit: Option<String>,
    precision: Option<usize>,
}

impl PredictionAccumulatorState {
    /// Create a new [prediction accumulator state](PredictionAccumulatorState).
    pub fn new() -> Self {
        Self {
            predictions: vec![],
            targets: vec![],
            current: None,
        }
    }

    /// Accumulate a batch of predictions and targets.
    pub fn accumulate(&mut self, preds: Tensor<2>, targets: Tensor<2, Bool>) {
        self.predictions.push(preds);
        self.targets.push(targets);
    }

    /// All accumulated predictions and targets, concatenated along the samples.
    pub fn tensors(&self) -> (Tensor<2>, Tensor<2, Bool>) {
        (
            Tensor::cat(self.predictions.clone(), 0),
            Tensor::cat(self.targets.clone(), 0),
        )
    }

    /// Record the value computed over the accumulated set and return the entry
    /// to log.
    pub fn compute(&mut self, value: f64, format: FormatOptions) -> SerializedEntry {
        self.current = Some(value);

        let serialized = NumericEntry::Value(value).serialize();

        let formatted_value = match format.precision {
            Some(precision) => format_float(value, precision),
            None => format!("{value}"),
        };

        // Rank-based metrics have no mathematically valid "batch" slice, so we render N/A
        let formatted = match format.unit {
            Some(unit) => format!("epoch {formatted_value} {unit} - batch N/A {unit}"),
            None => format!("epoch {formatted_value} - batch N/A"),
        };

        SerializedEntry::new(formatted, serialized)
    }

    /// Get the current metric value, when available.
    pub fn value(&self) -> Option<NumericEntry> {
        self.current.map(NumericEntry::Value)
    }

    /// Reset the state, freeing the accumulated tensors.
    pub fn reset(&mut self) {
        self.predictions.clear();
        self.targets.clear();
        self.current = None;
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.predictions.is_empty()
    }

    pub(crate) fn serialize_placeholder(&self, format: FormatOptions) -> SerializedEntry {
        SerializedEntry::not_available(format.unit.as_deref())
    }
}

impl Default for PredictionAccumulatorState {
    fn default() -> Self {
        Self::new()
    }
}

impl FormatOptions {
    /// Create the [formatting options](FormatOptions) with a name.
    pub fn new(name: MetricName) -> Self {
        Self {
            name: name.clone(),
            unit: None,
            precision: None,
        }
    }

    /// Specify the metric unit.
    pub fn unit(mut self, unit: &str) -> Self {
        self.unit = Some(unit.to_string());
        self
    }

    /// Specify the floating point precision.
    pub fn precision(mut self, precision: usize) -> Self {
        self.precision = Some(precision);
        self
    }

    /// Get the metric name.
    pub fn name(&self) -> &Arc<String> {
        &self.name
    }

    /// Get the metric unit.
    pub fn unit_value(&self) -> &Option<String> {
        &self.unit
    }

    /// Get the precision.
    pub fn precision_value(&self) -> Option<usize> {
        self.precision
    }
}

impl NumericMetricState {
    /// Create a new [numeric metric state](NumericMetricState).
    pub fn new() -> Self {
        Self {
            sum: 0.0,
            count: 0,
            current: f64::NAN,
            current_count: 0,
        }
    }

    /// Reset the state.
    pub fn reset(&mut self) {
        self.sum = 0.0;
        self.count = 0;
        self.current = f64::NAN;
        self.current_count = 0;
    }

    /// Update the state.
    ///
    /// `count` is the number of underlying units `value` is averaged over.
    /// This is typically the batch size (e.g. for accuracy, loss), but it can be any
    /// unit the metric's `value` is implicitly a rate over.
    pub fn update(&mut self, value: f64, count: usize) {
        self.sum += value * count as f64;
        self.count += count;
        self.current = value;
        self.current_count = count;
    }

    /// Compute the metric for the current update.
    pub fn compute_update(&self, format: FormatOptions) -> SerializedEntry {
        self.compute(format, false)
    }

    /// Compute the final metric for the accumulated global state.
    pub fn compute_final(&self, format: FormatOptions) -> SerializedEntry {
        self.compute(format, true)
    }

    fn compute(&self, format: FormatOptions, final_entry: bool) -> SerializedEntry {
        let value_current = self.current;
        let count = self.current_count;
        let value_running = self.sum / self.count as f64;
        // Numeric metric state is an aggregated value
        let serialized = if final_entry {
            NumericEntry::Final(value_running).serialize()
        } else {
            NumericEntry::Aggregated {
                aggregated_value: value_current,
                count,
            }
            .serialize()
        };

        let (formatted_current, formatted_running) = match format.precision {
            Some(precision) => (
                format_float(value_current, precision),
                format_float(value_running, precision),
            ),
            None => (format!("{value_current}"), format!("{value_running}")),
        };

        // TODO: naming inconsistent with RL.
        let formatted = match format.unit {
            Some(unit) => {
                format!("epoch {formatted_running} {unit} - batch {formatted_current} {unit}")
            }
            None => format!("epoch {formatted_running} - batch {formatted_current}"),
        };

        SerializedEntry::new(formatted, serialized)
    }

    /// Get the numeric value.
    pub fn current_value(&self) -> NumericEntry {
        NumericEntry::Aggregated {
            aggregated_value: self.current,
            count: self.current_count,
        }
    }

    /// Get the running aggregated value.
    pub fn running_value(&self) -> NumericEntry {
        NumericEntry::Aggregated {
            aggregated_value: self.sum / self.count as f64,
            count: self.count,
        }
    }

    /// Get the final aggregated value.
    pub fn final_value(&self) -> NumericEntry {
        NumericEntry::Final(self.sum / self.count as f64)
    }
}

impl Default for NumericMetricState {
    fn default() -> Self {
        Self::new()
    }
}

/// Accumulates confusion-matrix counts (TP, FP, FN) across an epoch.
///
/// Used by metrics derived from confusion-matrix ratios (precision, recall, etc.)
/// which must accumulate raw counts and compute the ratio over the summed counts.
/// Averaging the per-batch ratios (using [NumericMetricState]) is not equivalent;
/// it is biased whenever per-batch class support varies.
/// Accumulates confusion-matrix counts (TP, FP, FN) across an epoch using native tensors.
#[derive(Clone)]
pub struct ConfusionStatsState {
    // Epoch-level accumulated totals
    true_positive: Option<Tensor<1>>,
    false_positive: Option<Tensor<1>>,
    false_negative: Option<Tensor<1>>,
    running_count: usize,

    // Most recent batch snapshots (retained for the compute split)
    current_tp: Option<Tensor<1>>,
    current_fp: Option<Tensor<1>>,
    current_fn: Option<Tensor<1>>,
    current_count: usize,

    // Already computed values
    current_value: f64,
    running_value: f64,
}

impl Default for ConfusionStatsState {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfusionStatsState {
    /// Create a new [ConfusionStatsState].
    pub fn new() -> Self {
        Self {
            true_positive: None,
            false_positive: None,
            false_negative: None,
            running_count: 0,
            current_tp: None,
            current_fp: None,
            current_fn: None,
            current_count: 0,
            current_value: f64::NAN,
            running_value: f64::NAN,
        }
    }

    /// Reset the state.
    pub fn reset(&mut self) {
        self.true_positive = None;
        self.false_positive = None;
        self.false_negative = None;
        self.running_count = 0;
        self.current_tp = None;
        self.current_fp = None;
        self.current_fn = None;
        self.current_count = 0;
        self.current_value = f64::NAN;
        self.running_value = f64::NAN;
    }

    fn accumulate(running: &mut Option<Tensor<1>>, batch: Option<Tensor<1>>) {
        if let Some(b) = batch {
            *running = Some(match running.take() {
                Some(acc) => acc + b,
                None => b,
            });
        }
    }

    /// Accumulate running totals and capture the batch snapshot.
    pub fn update(
        &mut self,
        tp: Option<Tensor<1>>,
        fp: Option<Tensor<1>>,
        fn_: Option<Tensor<1>>,
        sample_size: usize,
    ) {
        // Accumulate running totals for global metric value
        Self::accumulate(&mut self.true_positive, tp.clone());
        Self::accumulate(&mut self.false_positive, fp.clone());
        Self::accumulate(&mut self.false_negative, fn_.clone());
        self.running_count += sample_size;

        // Retain current batch tensors for the subsequent compute pass
        self.current_tp = tp;
        self.current_fp = fp;
        self.current_fn = fn_;
        self.current_count = sample_size;
    }

    /// Compute the batch-level value and the running epoch-level value (from all accumulated counts)
    pub fn compute_update(
        &mut self,
        class_reduction: ClassReduction,
        format: FormatOptions,
        compute_fn: impl Fn(Option<Tensor<1>>, Option<Tensor<1>>, Option<Tensor<1>>) -> Tensor<1>,
    ) -> SerializedEntry {
        let current_tp = self.current_tp.take();
        let current_fp = self.current_fp.take();
        let current_fn = self.current_fn.take();

        // Compute batch-level value
        let batch_metric = compute_fn(current_tp, current_fp, current_fn);
        self.current_value = Self::class_average(batch_metric, class_reduction);

        // Compute epoch-level value from running totals
        let total_tp = self.true_positive.clone();
        let total_fp = self.false_positive.clone();
        let total_fn = self.false_negative.clone();

        let epoch_metric = compute_fn(total_tp, total_fp, total_fn);
        self.running_value = Self::class_average(epoch_metric, class_reduction);

        // Serialize and format
        let serialized = NumericEntry::Aggregated {
            aggregated_value: self.current_value,
            count: self.current_count,
        }
        .serialize();
        self.serialized_entry(format, serialized)
    }

    /// Compute the final metric for the accumulated global state.
    pub fn compute_final(&mut self, format: FormatOptions) -> SerializedEntry {
        self.serialized_entry(format, NumericEntry::Final(self.running_value).serialize())
    }

    fn serialized_entry(&mut self, format: FormatOptions, serialized: String) -> SerializedEntry {
        let (formatted_current, formatted_running) = match format.precision {
            Some(p) => (
                format_float(self.current_value, p),
                format_float(self.running_value, p),
            ),
            None => (
                format!("{}", self.current_value),
                format!("{}", self.running_value),
            ),
        };

        let formatted = match format.unit {
            Some(unit) => {
                format!("epoch {formatted_running} {unit} - batch {formatted_current} {unit}")
            }
            None => format!("epoch {formatted_running} - batch {formatted_current}"),
        };

        SerializedEntry::new(formatted, serialized)
    }

    fn class_average(mut metric: Tensor<1>, class_reduction: ClassReduction) -> f64 {
        use ClassReduction::{Macro, Micro};
        let avg = match class_reduction {
            Micro => metric,
            Macro => {
                if metric.clone().contains_nan().any().into_scalar() {
                    let mask = metric.clone().is_nan();
                    metric = metric
                        .clone()
                        .select(0, mask.bool_not().argwhere().squeeze_dim(1));
                }
                metric.mean()
            }
        };
        avg.into_scalar()
    }

    /// Get the current batch value.
    pub fn current_value(&self) -> Option<NumericEntry> {
        (!self.current_value.is_nan()).then(|| NumericEntry::Aggregated {
            aggregated_value: self.current_value,
            count: self.current_count,
        })
    }

    /// Get the running aggregated value.
    pub fn running_value(&self) -> Option<NumericEntry> {
        (!self.running_value.is_nan()).then(|| NumericEntry::Aggregated {
            aggregated_value: self.running_value,
            count: self.running_count,
        })
    }

    /// Get the final value of the metric.
    pub fn final_value(&self) -> NumericEntry {
        // The running value holds the epoch-level value from accumulated totals, which should hold
        // the correct final value after all batches have been processed
        NumericEntry::Final(self.running_value)
    }
}
