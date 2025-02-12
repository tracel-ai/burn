use crate::metric::{format_float, MetricEntry, Numeric, NumericEntry};

/// Useful utility to implement numeric metrics.
///
/// # Notes
///
/// The numeric metric store values inside floats.
/// Even if some metric are integers, their mean are floats.
pub struct NumericMetricState {
    sum: f64,
    count: usize,
    current: f64,
}

/// Formatting options for the [numeric metric state](NumericMetricState).
pub struct FormatOptions {
    name: String,
    unit: Option<String>,
    precision: Option<usize>,
}

impl FormatOptions {
    /// Create the [formatting options](FormatOptions) with a name.
    pub fn new<S: Into<String>>(name: S) -> Self {
        Self {
            name: name.into(),
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
}

impl NumericMetricState {
    /// Create a new [numeric metric state](NumericMetricState).
    pub fn new() -> Self {
        Self {
            sum: 0.0,
            count: 0,
            current: f64::NAN,
        }
    }

    /// Reset the state.
    pub fn reset(&mut self) {
        self.sum = 0.0;
        self.count = 0;
        self.current = f64::NAN;
    }

    /// Update the state.
    pub fn update(&mut self, value: f64, batch_size: usize, format: FormatOptions) -> MetricEntry {
        self.sum += value * batch_size as f64;
        self.count += batch_size;
        self.current = value;

        let value_current = value;
        let value_running = self.sum / self.count as f64;
        // Numeric metric state is an aggregated value
        let serialized = NumericEntry::Aggregated(value_current, batch_size).serialize();

        let (formatted_current, formatted_running) = match format.precision {
            Some(precision) => (
                format_float(value_current, precision),
                format_float(value_running, precision),
            ),
            None => (format!("{value_current}"), format!("{value_running}")),
        };

        let formatted = match format.unit {
            Some(unit) => {
                format!("epoch {formatted_running} {unit} - batch {formatted_current} {unit}")
            }
            None => format!("epoch {formatted_running} - batch {formatted_current}"),
        };

        MetricEntry::new(format.name, formatted, serialized)
    }
}

impl Numeric for NumericMetricState {
    fn value(&self) -> f64 {
        self.current
    }
}

impl Default for NumericMetricState {
    fn default() -> Self {
        Self::new()
    }
}
