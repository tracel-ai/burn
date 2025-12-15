use std::sync::Arc;

/// CPU Temperature metric
use super::MetricMetadata;
use crate::metric::{Metric, MetricAttributes, MetricName, Numeric, NumericEntry, SerializedEntry};
use systemstat::{Platform, System};

/// CPU Temperature in celsius degrees
#[derive(Clone)]
pub struct CpuTemperature {
    name: MetricName,
    temp_celsius: f32,
    sys: Arc<System>,
}

impl CpuTemperature {
    /// Creates a new CPU temp metric
    pub fn new() -> Self {
        let name = Arc::new("CPU Temperature".to_string());

        Self {
            name,
            temp_celsius: 0.,
            sys: Arc::new(System::new()),
        }
    }
}

impl Default for CpuTemperature {
    fn default() -> Self {
        CpuTemperature::new()
    }
}

impl Metric for CpuTemperature {
    type Input = ();

    fn update(&mut self, _item: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        match self.sys.cpu_temp() {
            Ok(temp) => self.temp_celsius = temp,
            Err(_) => self.temp_celsius = f32::NAN,
        }

        let formatted = match self.temp_celsius.is_nan() {
            true => format!("{}: NaN °C", self.name()),
            false => format!("{}: {:.2} °C", self.name(), self.temp_celsius),
        };
        let raw = format!("{:.2}", self.temp_celsius);

        SerializedEntry::new(formatted, raw)
    }

    fn clear(&mut self) {}

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        super::NumericAttributes {
            unit: Some("°C".to_string()),
            higher_is_better: false,
        }
        .into()
    }
}

impl Numeric for CpuTemperature {
    fn value(&self) -> NumericEntry {
        NumericEntry::Value(self.temp_celsius as f64)
    }

    fn running_value(&self) -> NumericEntry {
        NumericEntry::Value(self.temp_celsius as f64)
    }
}
