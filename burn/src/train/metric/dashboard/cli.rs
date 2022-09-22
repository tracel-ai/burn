use super::{DashboardMetricState, DashboardRenderer, TrainingProgress};
use indicatif::{MultiProgress, ProgressBar, ProgressState, ProgressStyle};
use std::{collections::HashMap, fmt::Write};

static MAX_REFRESH_RATE_MILLIS: u128 = 50;

pub struct CLIDashboardRenderer {
    name: String,
    pb_epoch: ProgressBar,
    pb_iteration: ProgressBar,
    last_update: std::time::Instant,
    progress: TrainingProgress,
    metric_train: HashMap<String, String>,
    metric_valid: HashMap<String, String>,
    metric_train_numeric: HashMap<String, String>,
    metric_valid_numeric: HashMap<String, String>,
}

impl DashboardRenderer for CLIDashboardRenderer {
    fn update_train(&mut self, state: DashboardMetricState) {
        match state {
            DashboardMetricState::Generic(state) => {
                self.metric_train.insert(state.name(), state.pretty());
            }
            DashboardMetricState::Numeric(state, _value) => {
                self.metric_train_numeric
                    .insert(state.name(), state.pretty());
            }
        };
    }

    fn update_valid(&mut self, state: DashboardMetricState) {
        match state {
            DashboardMetricState::Generic(state) => {
                self.metric_valid.insert(state.name(), state.pretty());
            }
            DashboardMetricState::Numeric(state, _value) => {
                self.metric_valid_numeric
                    .insert(state.name(), state.pretty());
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
    pub fn new(name: &str) -> Self {
        let pb = MultiProgress::new();
        let pb_epoch = ProgressBar::new(0);
        let pb_iteration = ProgressBar::new(0);

        let pb_iteration = pb.add(pb_iteration);
        let pb_epoch = pb.add(pb_epoch);

        Self {
            name: name.to_string(),
            pb_epoch,
            pb_iteration,
            last_update: std::time::Instant::now(),
            progress: TrainingProgress::none(),
            metric_train: HashMap::new(),
            metric_valid: HashMap::new(),
            metric_train_numeric: HashMap::new(),
            metric_valid_numeric: HashMap::new(),
        }
    }

    fn register_template_metrics(&self, template: String) -> String {
        let mut template = template;
        let mut metrics_keys = Vec::new();

        for (name, metric) in self.metric_train.iter() {
            metrics_keys.push(format!("  - Train {}: {}", name, metric));
        }
        for (name, metric) in self.metric_train_numeric.iter() {
            metrics_keys.push(format!("  - Train {}: {}", name, metric));
        }
        for (name, metric) in self.metric_valid.iter() {
            metrics_keys.push(format!("  - Valid {}: {}", name, metric));
        }
        for (name, metric) in self.metric_valid_numeric.iter() {
            metrics_keys.push(format!("  - Valid {}: {}", name, metric));
        }

        if !metrics_keys.is_empty() {
            let metrics_template = metrics_keys.join("\n");
            template += format!("{}\n{}\n", METRICS_TAG, metrics_template).as_str();
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

        let bar = "[{wide_bar:.cyan/blue}] ({eta})";
        template += format!("  - {} {}", progress, bar).as_str();
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

        let template = format!("{}\n  - Name: {}\n", GENERAL_TAG, self.name);
        let template = self.register_template_metrics(template);
        let template = template + format!("{}\n", PROGRESS_TAG).as_str();

        let template = self.register_template_progress("iteration", template);
        let style_iteration = ProgressStyle::with_template(&template).unwrap();
        let style_iteration = self.register_style_progress(
            "iteration",
            style_iteration,
            format!("{}", self.progress.iteration),
        );

        let template = self.register_template_progress("epoch", "".to_string());
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
        self.pb_epoch.set_position(self.progress.epoch as u64);
        self.pb_epoch.set_length(self.progress.epoch_total as u64);

        self.last_update = std::time::Instant::now();
    }

    pub fn register_key_item(
        &self,
        key: &'static str,
        style: ProgressStyle,
        name: String,
        formatted: String,
    ) -> ProgressStyle {
        style.with_key(key, move |_state: &ProgressState, w: &mut dyn Write| {
            write!(w, "{}: {}", name, formatted).unwrap()
        })
    }
}

static GENERAL_TAG: &str = "[General]";
static METRICS_TAG: &str = "[Metrics]";
static PROGRESS_TAG: &str = "[Progress]";
