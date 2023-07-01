use super::{DashboardMetricState, DashboardRenderer, TextPlot, TrainingProgress};
use indicatif::{MultiProgress, ProgressBar, ProgressState, ProgressStyle};
use std::{collections::HashMap, fmt::Write};

static MAX_REFRESH_RATE_MILLIS: u128 = 250;

/// The CLI dashboard renderer.
pub struct CLIDashboardRenderer {
    pb_epoch: ProgressBar,
    pb_iteration: ProgressBar,
    last_update: std::time::Instant,
    progress: TrainingProgress,
    metric_train: HashMap<String, String>,
    metric_valid: HashMap<String, String>,
    metric_both_plot: HashMap<String, TextPlot>,
    metric_train_plot: HashMap<String, TextPlot>,
    metric_valid_plot: HashMap<String, TextPlot>,
}

impl Default for CLIDashboardRenderer {
    fn default() -> Self {
        CLIDashboardRenderer::new()
    }
}

impl Drop for CLIDashboardRenderer {
    fn drop(&mut self) {
        self.pb_iteration.finish();
        self.pb_epoch.finish();
    }
}

impl DashboardRenderer for CLIDashboardRenderer {
    fn update_train(&mut self, state: DashboardMetricState) {
        match state {
            DashboardMetricState::Generic(state) => {
                self.metric_train.insert(state.name, state.formatted);
            }
            DashboardMetricState::Numeric(state, value) => {
                let name = &state.name;
                self.metric_train.insert(name.clone(), state.formatted);

                if let Some(mut plot) = self.text_plot_in_both(name) {
                    plot.update_train(value as f32);
                    self.metric_both_plot.insert(name.clone(), plot);
                    return;
                }

                if let Some(plot) = self.metric_train_plot.get_mut(name) {
                    plot.update_train(value as f32);
                } else {
                    let mut plot = TextPlot::new();
                    plot.update_train(value as f32);
                    self.metric_train_plot.insert(state.name, plot);
                }
            }
        };
    }

    fn update_valid(&mut self, state: DashboardMetricState) {
        match state {
            DashboardMetricState::Generic(state) => {
                self.metric_valid.insert(state.name, state.formatted);
            }
            DashboardMetricState::Numeric(state, value) => {
                let name = &state.name;
                self.metric_valid.insert(name.clone(), state.formatted);

                if let Some(mut plot) = self.text_plot_in_both(name) {
                    plot.update_valid(value as f32);
                    self.metric_both_plot.insert(name.clone(), plot);
                    return;
                }

                if let Some(plot) = self.metric_valid_plot.get_mut(name) {
                    plot.update_valid(value as f32);
                } else {
                    let mut plot = TextPlot::new();
                    plot.update_valid(value as f32);
                    self.metric_valid_plot.insert(state.name, plot);
                }
            }
        };
    }

    fn render_train(&mut self, item: TrainingProgress) {
        self.progress = item;
        self.render();
    }

    fn render_valid(&mut self, item: TrainingProgress) {
        self.progress = item;
        self.render();
    }
}

impl CLIDashboardRenderer {
    /// Create a new CLI dashboard renderer.
    pub fn new() -> Self {
        let pb = MultiProgress::new();
        let pb_epoch = ProgressBar::new(0);
        let pb_iteration = ProgressBar::new(0);

        let pb_iteration = pb.add(pb_iteration);
        let pb_epoch = pb.add(pb_epoch);

        Self {
            pb_epoch,
            pb_iteration,
            last_update: std::time::Instant::now(),
            progress: TrainingProgress::none(),
            metric_train: HashMap::new(),
            metric_valid: HashMap::new(),
            metric_both_plot: HashMap::new(),
            metric_train_plot: HashMap::new(),
            metric_valid_plot: HashMap::new(),
        }
    }

    fn text_plot_in_both(&mut self, key: &str) -> Option<TextPlot> {
        if let Some(plot) = self.metric_both_plot.remove(key) {
            return Some(plot);
        }
        if self.metric_train_plot.contains_key(key) && self.metric_valid_plot.contains_key(key) {
            let plot_train = self.metric_train_plot.remove(key).unwrap();
            let plot_valid = self.metric_valid_plot.remove(key).unwrap();

            return Some(plot_train.merge(plot_valid));
        }

        None
    }

    fn register_template_plots(&self, template: String) -> String {
        let mut template = template;
        let mut metrics_keys = Vec::new();

        for (name, metric) in self.metric_both_plot.iter() {
            metrics_keys.push(format!(
                "  - {} RED: train | BLUE: valid \n{}",
                name,
                metric.render()
            ));
        }
        for (name, metric) in self.metric_train_plot.iter() {
            metrics_keys.push(format!("  - Train {}: \n{}", name, metric.render()));
        }
        for (name, metric) in self.metric_valid_plot.iter() {
            metrics_keys.push(format!("  - Valid {}: \n{}", name, metric.render()));
        }

        if !metrics_keys.is_empty() {
            let metrics_template = metrics_keys.join("\n");
            template += format!("{PLOTS_TAG}\n{metrics_template}\n").as_str();
        }

        template
    }
    fn register_template_metrics(&self, template: String) -> String {
        let mut template = template;
        let mut metrics_keys = Vec::new();

        for (name, metric) in self.metric_train.iter() {
            metrics_keys.push(format!("  - Train {name}: {metric}"));
        }
        for (name, metric) in self.metric_valid.iter() {
            metrics_keys.push(format!("  - Valid {name}: {metric}"));
        }

        if !metrics_keys.is_empty() {
            let metrics_template = metrics_keys.join("\n");
            template += format!("{METRICS_TAG}\n{metrics_template}\n").as_str();
        }

        template
    }

    fn register_style_progress(
        &self,
        name: &'static str,
        style: ProgressStyle,
        value: String,
    ) -> ProgressStyle {
        self.register_key_item(name, style, name.to_string(), value)
    }

    fn register_template_progress(&self, progress: &str, template: String) -> String {
        let mut template = template;

        let bar = "[{wide_bar:.cyan/blue}]";
        template += format!("  - {progress} {bar}").as_str();
        template
    }

    fn render(&mut self) {
        if std::time::Instant::now()
            .duration_since(self.last_update)
            .as_millis()
            < MAX_REFRESH_RATE_MILLIS
        {
            return;
        }

        let template = self.register_template_plots(String::default());
        let template = self.register_template_metrics(template);
        let template = template
            + format!(
                "\n{}\n  - Iteration {} Epoch {}/{}\n",
                PROGRESS_TAG,
                self.progress.iteration,
                self.progress.epoch,
                self.progress.epoch_total
            )
            .as_str();

        let template = self.register_template_progress("iteration", template);
        let style_iteration = ProgressStyle::with_template(&template).unwrap();
        let style_iteration = self.register_style_progress(
            "iteration",
            style_iteration,
            format!("{}", self.progress.iteration),
        );

        let template = self.register_template_progress("epoch    ", String::default());
        let style_epoch = ProgressStyle::with_template(&template).unwrap();
        let style_epoch =
            self.register_style_progress("epoch", style_epoch, format!("{}", self.progress.epoch));

        self.pb_iteration
            .set_style(style_iteration.progress_chars("#>-"));
        self.pb_iteration
            .set_position(self.progress.progress.items_processed as u64);
        self.pb_iteration
            .set_length(self.progress.progress.items_total as u64);

        self.pb_epoch.set_style(style_epoch.progress_chars("#>-"));
        self.pb_epoch.set_position(self.progress.epoch as u64 - 1);
        self.pb_epoch.set_length(self.progress.epoch_total as u64);

        self.last_update = std::time::Instant::now();
    }

    /// Registers a new metric to be displayed.
    pub fn register_key_item(
        &self,
        key: &'static str,
        style: ProgressStyle,
        name: String,
        formatted: String,
    ) -> ProgressStyle {
        style.with_key(key, move |_state: &ProgressState, w: &mut dyn Write| {
            write!(w, "{name}: {formatted}").unwrap()
        })
    }
}

static METRICS_TAG: &str = "[Metrics]";
static PLOTS_TAG: &str = "[Plots]";
static PROGRESS_TAG: &str = "[Progress]";
