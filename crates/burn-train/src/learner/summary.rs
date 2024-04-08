use core::cmp::Ordering;
use std::{fmt::Display, path::Path};

use crate::{
    logger::FileMetricLogger,
    metric::store::{Aggregate, EventStore, LogEventStore, Split},
};

/// Contains the metric value at a given time.
pub struct MetricEntry {
    /// The step at which the metric was recorded (i.e., epoch).
    pub step: usize,
    /// The metric value.
    pub value: f64,
}

/// Contains the summary of recorded values for a given metric.
pub struct MetricSummary {
    /// The metric name.
    pub name: String,
    /// The metric entries.
    pub entries: Vec<MetricEntry>,
}

impl MetricSummary {
    fn new<E: EventStore>(
        event_store: &mut E,
        metric: &str,
        split: Split,
        num_epochs: usize,
    ) -> Self {
        let entries = (1..num_epochs)
            .into_iter()
            .filter_map(|epoch| {
                if let Some(value) = event_store.find_metric(metric, epoch, Aggregate::Mean, split)
                {
                    Some(MetricEntry { step: epoch, value })
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        Self {
            name: metric.to_string(),
            entries,
        }
    }
}

/// Contains the summary of recorded metrics for the training and validation steps.
pub struct SummaryMetrics {
    /// Training metrics summary.
    pub train: Vec<MetricSummary>,
    /// Validation metrics summary.
    pub valid: Vec<MetricSummary>,
}

/// Detailed training summary.
pub struct LearnerSummary {
    /// The number of epochs completed.
    pub epochs: usize,
    /// The summary of recorded metrics during training.
    pub metrics: SummaryMetrics,
}

impl LearnerSummary {
    /// Creates a new learner summary for the specified metrics.
    ///
    /// # Arguments
    ///
    /// * `directory` - The directory containing the training artifacts (checkpoints and logs).
    /// * `metrics` - The list of metrics to collect for the summary.
    pub fn new(directory: &str, metrics: &[&str]) -> Self {
        if !Path::new(directory).exists() {
            panic!("Artifact directory does not exist at: {}", directory);
        }
        let mut event_store = LogEventStore::default();

        let train_logger = FileMetricLogger::new(format!("{directory}/train").as_str());
        let valid_logger = FileMetricLogger::new(format!("{directory}/valid").as_str());

        // Number of recorded epochs
        let epochs = train_logger.epochs();

        event_store.register_logger_train(train_logger);
        event_store.register_logger_valid(valid_logger);

        let train_summary = metrics
            .iter()
            .map(|metric| MetricSummary::new(&mut event_store, metric, Split::Train, epochs))
            .collect::<Vec<_>>();

        let valid_summary = metrics
            .iter()
            .map(|metric| MetricSummary::new(&mut event_store, metric, Split::Valid, epochs))
            .collect::<Vec<_>>();

        Self {
            epochs,
            metrics: SummaryMetrics {
                train: train_summary,
                valid: valid_summary,
            },
        }
    }
}

impl Display for LearnerSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Compute the max length for each column
        let split_train = "Train";
        let split_valid = "Valid";
        let max_split_len = "Split".len().max(split_train.len()).max(split_valid.len());
        let mut max_metric_len = "Metric".len();
        for metric in self.metrics.train.iter() {
            max_metric_len = max_metric_len.max(metric.name.len());
        }
        for metric in self.metrics.valid.iter() {
            max_metric_len = max_metric_len.max(metric.name.len());
        }

        // Summary header
        writeln!(
            f,
            "{:=>width_symbol$} Learner Summary {:=>width_symbol$}\nTotal Epochs: {epochs}\n\n",
            "",
            "",
            width_symbol = 24,
            epochs = self.epochs,
        )?;

        // Metrics table header
        writeln!(
            f,
            "| {:<width_split$} | {:<width_metric$} | Min.     | Epoch    | Max.     | Epoch    |\n|{:->width_split$}--|{:->width_metric$}--|----------|----------|----------|----------|",
            "Split", "Metric", "", "",
            width_split = max_split_len,
            width_metric = max_metric_len,
        )?;

        // Table entries
        fn cmp_f64(a: &f64, b: &f64) -> Ordering {
            match (a.is_nan(), b.is_nan()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                _ => a.partial_cmp(b).unwrap(),
            }
        }

        let mut write_metrics_summary = |metrics: &[MetricSummary],
                                         split: &str|
         -> std::fmt::Result {
            for metric in metrics.iter() {
                if metric.entries.is_empty() {
                    continue; // skip metrics with no recorded values
                }

                // Compute the min & max for each metric
                let metric_min = metric
                    .entries
                    .iter()
                    .min_by(|a, b| cmp_f64(&a.value, &b.value))
                    .unwrap();
                let metric_max = metric
                    .entries
                    .iter()
                    .max_by(|a, b| cmp_f64(&a.value, &b.value))
                    .unwrap();

                writeln!(
                    f,
                    "| {:<width_split$} | {:<width_metric$} | {:<9.3?}| {:<9?}| {:<9.3?}| {:<9.3?}|",
                    split,
                    metric.name,
                    metric_min.value,
                    metric_min.step,
                    metric_max.value,
                    metric_max.step,
                    width_split = max_split_len,
                    width_metric = max_metric_len,
                )?;
            }

            Ok(())
        };

        write_metrics_summary(&self.metrics.train, split_train)?;
        write_metrics_summary(&self.metrics.valid, split_valid)?;

        Ok(())
    }
}
