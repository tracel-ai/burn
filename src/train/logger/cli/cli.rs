use crate::train::{
    logger::{LogItem, Logger},
    metric::{LossMetric, Metric, MetricStateDyn},
};
use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use std::fmt::Write;

pub struct CLILogger<T> {
    metrics: Vec<Box<dyn Metric<T>>>,
    name: String,
    pb: ProgressBar,
}

impl<T> Logger<T> for CLILogger<T>
where
    LossMetric: Metric<T>,
{
    fn log(&mut self, item: LogItem<T>) {
        let metrics = self.update_metrics(&item);

        let template = format!("{}\n  - Name: {}\n", GENERAL_TAG, self.name);
        let template = self.register_template_metrics(&metrics, template);
        let template = self.register_template_progress(&item, template);

        let style = ProgressStyle::with_template(&template).unwrap();
        let style = self.register_style_progress(&item, style);

        if self.pb.length() == Some(0) {
            self.pb.println("\n\n");
        }

        self.pb.set_style(style.progress_chars("#>-"));
        self.pb.set_position(item.progress.items_processed as u64);
        self.pb.set_length(item.progress.items_total as u64);
        self.pb.tick();
    }

    fn clear(&mut self) {
        self.pb.finish();
        self.pb = ProgressBar::new(0);

        for metric in &mut self.metrics {
            metric.clear();
        }
    }
}

impl<T> CLILogger<T> {
    pub fn new(metrics: Vec<Box<dyn Metric<T>>>, name: String) -> Self {
        Self {
            metrics,
            name,
            pb: ProgressBar::new(0),
        }
    }

    pub fn update_metrics(&mut self, item: &LogItem<T>) -> Vec<MetricStateDyn> {
        let mut metrics_result = Vec::with_capacity(self.metrics.len());

        for metric in &mut self.metrics {
            metrics_result.push(metric.update(&item.item));
        }

        metrics_result
    }

    pub fn register_template_progress(&self, item: &LogItem<T>, template: String) -> String {
        let mut template = template;
        let mut progress = Vec::new();

        if let Some(_) = &item.epoch {
            progress.push("  - {epoch}");
        }

        progress.push("  - {iteration} [{wide_bar:.cyan/blue}] ({eta})  ");

        if progress.len() > 0 {
            let progress = progress.join("\n");
            template = template + format!("{}\n{}\n", PROGRESS_TAG, progress).as_str();
        }

        template
    }

    pub fn register_template_metrics(
        &self,
        metrics: &Vec<MetricStateDyn>,
        template: String,
    ) -> String {
        let mut template = template;
        let mut metrics_keys = Vec::new();

        for metric in metrics {
            metrics_keys.push(format!("  - {}: {}", metric.name(), metric.pretty()));
        }

        if metrics.len() > 0 {
            let metrics_template = metrics_keys.join("\n");
            template = template + format!("{}\n{}\n", METRICS_TAG, metrics_template).as_str();
        }

        template
    }

    pub fn register_style_progress(
        &self,
        item: &LogItem<T>,
        style: ProgressStyle,
    ) -> ProgressStyle {
        let mut style = self.register_key_item(
            "iteration",
            style,
            String::from("Iteration"),
            format!("{}", item.iteration.unwrap_or(0)),
        );

        if let Some(epoch) = item.epoch {
            let formatted = match item.epoch_total {
                Some(total) => format!("{}/{}", epoch, total),
                None => format!("{}", epoch),
            };
            let name = String::from("Epoch");

            style = self.register_key_item("epoch", style, name, formatted);
        }

        style
    }

    pub fn register_style_metrics(
        &self,
        items: &Vec<MetricStateDyn>,
        style: ProgressStyle,
    ) -> ProgressStyle {
        let mut style = style;

        for (i, result) in items.iter().enumerate() {
            style = match i {
                0 => self.register_single_metric_result(METRIC_0, style, result),
                1 => self.register_single_metric_result(METRIC_1, style, result),
                2 => self.register_single_metric_result(METRIC_2, style, result),
                3 => self.register_single_metric_result(METRIC_3, style, result),
                4 => self.register_single_metric_result(METRIC_4, style, result),
                5 => self.register_single_metric_result(METRIC_5, style, result),
                6 => self.register_single_metric_result(METRIC_6, style, result),
                7 => self.register_single_metric_result(METRIC_7, style, result),
                8 => self.register_single_metric_result(METRIC_8, style, result),
                9 => self.register_single_metric_result(METRIC_9, style, result),
                _ => panic!("Only support 10 metrics"),
            };
        }

        style
    }

    pub fn register_single_metric_result(
        &self,
        key: &'static str,
        style: ProgressStyle,
        metric_result: &MetricStateDyn,
    ) -> ProgressStyle {
        let formatted = metric_result.pretty();
        let name = metric_result.name();

        self.register_key_item(key, style, name, formatted)
    }

    pub fn register_key_item(
        &self,
        key: &'static str,
        style: ProgressStyle,
        name: String,
        formatted: String,
    ) -> ProgressStyle {
        let style = style.with_key(key, move |_state: &ProgressState, w: &mut dyn Write| {
            write!(w, "{}: {}", name, formatted).unwrap()
        });

        style
    }
}

static GENERAL_TAG: &str = "[General]";
static METRICS_TAG: &str = "[Metrics]";
static PROGRESS_TAG: &str = "[Progress]";

static METRIC_0: &str = "metric0";
static METRIC_1: &str = "metric1";
static METRIC_2: &str = "metric2";
static METRIC_3: &str = "metric3";
static METRIC_4: &str = "metric4";
static METRIC_5: &str = "metric5";
static METRIC_6: &str = "metric6";
static METRIC_7: &str = "metric7";
static METRIC_8: &str = "metric8";
static METRIC_9: &str = "metric9";
