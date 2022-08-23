use super::{LogItem, Logger};
use crate::train::metric::{LossMetric, RunningMetric};
use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use std::fmt::Write;

pub enum CLIMetric {
    Loss(LossMetric),
    Epoch,
}

pub struct CLILogger {
    metrics: Vec<CLIMetric>,
    name: String,
    pb: ProgressBar,
}

impl CLILogger {
    pub fn new(metrics: Vec<CLIMetric>, name: String) -> Self {
        Self {
            metrics,
            name,
            pb: ProgressBar::new(1),
        }
    }
}

impl<T> Logger<T> for CLILogger
where
    LossMetric: RunningMetric<T>,
{
    fn log(&mut self, item: LogItem<T>) {
        let mut template = format!("Task     : {}\n", self.name);
        let mut metrics = Vec::new();
        let mut progress = Vec::new();

        for metric in &self.metrics {
            match metric {
                CLIMetric::Loss(_) => {
                    metrics.push("{loss}");
                }
                CLIMetric::Epoch => {
                    progress.push("{epoch}");
                }
            }
        }
        if metrics.len() > 0 {
            let metrics = metrics.join(" - ");
            template = template + format!("Metrics  : {}\n", &metrics).as_str();
        }

        progress.push("{iteration} [{wide_bar:.cyan/blue}] ({eta})");

        if progress.len() > 0 {
            let progress = progress.join(" - ");
            template = template + format!("Progress : {}\n", &progress).as_str();
        }

        let mut style = ProgressStyle::with_template(&template).unwrap();
        style = style.with_key(
            "iteration",
            move |_state: &ProgressState, w: &mut dyn Write| {
                write!(w, "Iteration: {}", item.iteration).unwrap()
            },
        );
        for metric in &mut self.metrics {
            match metric {
                CLIMetric::Loss(metric) => {
                    let loss = metric.update(&item.item);
                    style =
                        style.with_key("loss", move |_state: &ProgressState, w: &mut dyn Write| {
                            write!(w, "{}", loss).unwrap()
                        });
                }
                CLIMetric::Epoch => {
                    let epoch = item.epoch.unwrap().clone();
                    let epoch_total = item.epoch_total.unwrap().clone();
                    style = style.with_key(
                        "epoch",
                        move |_state: &ProgressState, w: &mut dyn Write| {
                            write!(w, "Epoch {}/{}", epoch, epoch_total).unwrap()
                        },
                    );
                }
            }
        }
        self.pb.set_style(style.progress_chars("#>-"));
        self.pb.set_position(item.iteration as u64);
        self.pb.set_length(item.iteration_total as u64);
    }

    fn clear(&mut self) {
        self.pb.println("");
        self.pb.println("");
        self.pb.finish();
        self.pb = ProgressBar::new(1);

        for metric in &mut self.metrics {
            match metric {
                CLIMetric::Loss(metric) => metric.clear(),
                CLIMetric::Epoch => {}
            }
        }
    }
}
