use super::{LogItem, Logger};
use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use std::fmt::Write;

pub enum CLIMetric {
    Loss,
    Epoch,
}

pub struct CLILogger {
    metrics: Vec<CLIMetric>,
    name: String,
    pb: ProgressBar,
    last_epoch: usize,
}

impl CLILogger {
    pub fn new(metrics: Vec<CLIMetric>, name: String) -> Self {
        Self {
            metrics,
            name,
            pb: ProgressBar::new(1),
            last_epoch: 0,
        }
    }
}

impl<T> Logger<T> for CLILogger {
    fn log(&mut self, item: LogItem<T>) {
        let mut template = "{iteration} ".to_string();
        for metric in &self.metrics {
            match metric {
                CLIMetric::Loss => {
                    template = template + "{loss} ";
                }
                CLIMetric::Epoch => {
                    template = template + "{epoch} ";
                }
            }
        }
        template = template + "[{elapsed_precise}] [{wide_bar:.cyan/blue}] ({eta})";
        let mut style = ProgressStyle::with_template(&template).unwrap();
        style = style.with_key(
            "iteration",
            move |_state: &ProgressState, w: &mut dyn Write| {
                write!(w, "iteration {}", item.iteration).unwrap()
            },
        );
        for metric in &self.metrics {
            match metric {
                CLIMetric::Loss => {
                    style =
                        style.with_key("loss", move |_state: &ProgressState, w: &mut dyn Write| {
                            write!(w, "No loss :(").unwrap()
                        });
                }
                CLIMetric::Epoch => {
                    let epoch = item.epoch.unwrap().clone();
                    let epoch_total = item.epoch_total.unwrap().clone();
                    style = style.with_key(
                        "epoch",
                        move |_state: &ProgressState, w: &mut dyn Write| {
                            write!(w, "epoch {}/{}", epoch, epoch_total).unwrap()
                        },
                    );
                }
            }
        }
        match item.epoch {
            Some(epoch) => {
                if self.last_epoch < epoch {
                    self.pb.finish();
                    self.pb = ProgressBar::new(item.iteration_total as u64);
                    self.pb.println(format!("{}", self.name));
                    self.last_epoch = epoch;
                }
            }
            None => {}
        };

        self.pb.set_style(style.progress_chars("#>-"));
        self.pb.set_position(item.iteration as u64);
        self.pb.set_length(item.iteration_total as u64);
    }
}
