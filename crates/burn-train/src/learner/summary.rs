use core::cmp::Ordering;
use std::{
    fmt::Display,
    path::{Path, PathBuf},
};

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
    ) -> Option<Self> {
        let entries = (1..=num_epochs)
            .filter_map(|epoch| {
                event_store
                    .find_metric(metric, epoch, Aggregate::Mean, split)
                    .map(|value| MetricEntry { step: epoch, value })
            })
            .collect::<Vec<_>>();

        if entries.is_empty() {
            None
        } else {
            Some(Self {
                name: metric.to_string(),
                entries,
            })
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
    /// The model name (only recorded within the learner).
    pub(crate) model: Option<String>,
}

impl LearnerSummary {
    /// Creates a new learner summary for the specified metrics.
    ///
    /// # Arguments
    ///
    /// * `directory` - The directory containing the training artifacts (checkpoints and logs).
    /// * `metrics` - The list of metrics to collect for the summary.
    pub fn new<S: AsRef<str>>(directory: impl AsRef<Path>, metrics: &[S]) -> Result<Self, String> {
        let directory = directory.as_ref();
        if !directory.exists() {
            return Err(format!(
                "Artifact directory does not exist at: {}",
                directory.display()
            ));
        }
        let train_dir = directory.join("train");
        let valid_dir = directory.join("valid");
        if !train_dir.exists() & !valid_dir.exists() {
            return Err(format!(
                "No training or validation artifacts found at: {}",
                directory.display()
            ));
        }

        let mut event_store = LogEventStore::default();

        let train_logger = FileMetricLogger::new_train(train_dir.to_str().unwrap());
        let valid_logger = FileMetricLogger::new_train(valid_dir.to_str().unwrap());

        // Number of recorded epochs
        let epochs = train_logger.epochs();

        event_store.register_logger_train(train_logger);
        event_store.register_logger_valid(valid_logger);

        let train_summary = metrics
            .iter()
            .filter_map(|metric| {
                MetricSummary::new(&mut event_store, metric.as_ref(), Split::Train, epochs)
            })
            .collect::<Vec<_>>();

        let valid_summary = metrics
            .iter()
            .filter_map(|metric| {
                MetricSummary::new(&mut event_store, metric.as_ref(), Split::Valid, epochs)
            })
            .collect::<Vec<_>>();

        Ok(Self {
            epochs,
            metrics: SummaryMetrics {
                train: train_summary,
                valid: valid_summary,
            },
            model: None,
        })
    }

    pub(crate) fn with_model(mut self, name: String) -> Self {
        self.model = Some(name);
        self
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
            "{:=>width_symbol$} Learner Summary {:=>width_symbol$}",
            "",
            "",
            width_symbol = 24,
        )?;

        if let Some(model) = &self.model {
            writeln!(f, "Model:\n{model}")?;
        }
        writeln!(f, "Total Epochs: {epochs}\n\n", epochs = self.epochs)?;

        // Metrics table header
        writeln!(
            f,
            "| {:<width_split$} | {:<width_metric$} | Min.     | Epoch    | Max.     | Epoch    |\n|{:->width_split$}--|{:->width_metric$}--|----------|----------|----------|----------|",
            "Split",
            "Metric",
            "",
            "",
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

        fn fmt_val(val: f64) -> String {
            if val < 1e-2 {
                // Use scientific notation for small values which would otherwise be truncated
                format!("{val:<9.3e}")
            } else {
                format!("{val:<9.3}")
            }
        }

        let mut write_metrics_summary =
            |metrics: &[MetricSummary], split: &str| -> std::fmt::Result {
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
                        "| {:<width_split$} | {:<width_metric$} | {}| {:<9?}| {}| {:<9?}|",
                        split,
                        metric.name,
                        fmt_val(metric_min.value),
                        metric_min.step,
                        fmt_val(metric_max.value),
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

pub(crate) struct LearnerSummaryConfig {
    pub(crate) directory: PathBuf,
    pub(crate) metrics: Vec<String>,
}

impl LearnerSummaryConfig {
    pub fn init(&self) -> Result<LearnerSummary, String> {
        LearnerSummary::new(&self.directory, &self.metrics[..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic = "Summary artifacts should exist"]
    fn test_artifact_dir_should_exist() {
        let dir = "/tmp/learner-summary-not-found";
        let _summary = LearnerSummary::new(dir, &["Loss"]).expect("Summary artifacts should exist");
    }

    #[test]
    #[should_panic = "Summary artifacts should exist"]
    fn test_train_valid_artifacts_should_exist() {
        let dir = "/tmp/test-learner-summary-empty";
        std::fs::create_dir_all(dir).ok();
        let _summary = LearnerSummary::new(dir, &["Loss"]).expect("Summary artifacts should exist");
    }

    #[test]
    fn test_summary_should_be_empty() {
        let dir = Path::new("/tmp/test-learner-summary-empty-metrics");
        std::fs::create_dir_all(dir).unwrap();
        std::fs::create_dir_all(dir.join("train/epoch-1")).unwrap();
        std::fs::create_dir_all(dir.join("valid/epoch-1")).unwrap();
        let summary = LearnerSummary::new(dir.to_str().unwrap(), &["Loss"])
            .expect("Summary artifacts should exist");

        assert_eq!(summary.epochs, 1);

        assert_eq!(summary.metrics.train.len(), 0);
        assert_eq!(summary.metrics.valid.len(), 0);

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn test_summary_should_be_collected() {
        let dir = Path::new("/tmp/test-learner-summary");
        let train_dir = dir.join("train/epoch-1");
        let valid_dir = dir.join("valid/epoch-1");
        std::fs::create_dir_all(dir).unwrap();
        std::fs::create_dir_all(&train_dir).unwrap();
        std::fs::create_dir_all(&valid_dir).unwrap();

        std::fs::write(train_dir.join("Loss.log"), "1.0\n2.0").expect("Unable to write file");
        std::fs::write(valid_dir.join("Loss.log"), "1.0").expect("Unable to write file");

        let summary = LearnerSummary::new(dir.to_str().unwrap(), &["Loss"])
            .expect("Summary artifacts should exist");

        assert_eq!(summary.epochs, 1);

        // Only Loss metric
        assert_eq!(summary.metrics.train.len(), 1);
        assert_eq!(summary.metrics.valid.len(), 1);

        // Aggregated train metric entries for 1 epoch
        let train_metric = &summary.metrics.train[0];
        assert_eq!(train_metric.name, "Loss");
        assert_eq!(train_metric.entries.len(), 1);
        let entry = &train_metric.entries[0];
        assert_eq!(entry.step, 1); // epoch = 1
        assert_eq!(entry.value, 1.5); // (1 + 2) / 2

        // Aggregated valid metric entries for 1 epoch
        let valid_metric = &summary.metrics.valid[0];
        assert_eq!(valid_metric.name, "Loss");
        assert_eq!(valid_metric.entries.len(), 1);
        let entry = &valid_metric.entries[0];
        assert_eq!(entry.step, 1); // epoch = 1
        assert_eq!(entry.value, 1.0);

        std::fs::remove_dir_all(dir).unwrap();
    }
}
