/// CPU Temperature metric
use super::{MetricMetadata, Numeric};
use crate::metric::{Metric, MetricEntry};
use systemstat::{Platform, System};

/// CPU Temperature in celsius degrees
pub struct CpuTemperature {
    temp_celsius: f32,
    sys: System,
}

impl CpuTemperature {
    /// Creates a new CPU temp metric
    pub fn new() -> Self {
        Self {
            temp_celsius: 0.,
            sys: System::new(),
        }
    }
}

impl Default for CpuTemperature {
    fn default() -> Self {
        CpuTemperature::new()
    }
}

impl Metric for CpuTemperature {
    const NAME: &'static str = "CPU Temperature";

    type Input = ();

    fn update(&mut self, _item: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        match self.sys.cpu_temp() {
            Ok(temp) => self.temp_celsius = temp,
            Err(_) => self.temp_celsius = f32::NAN,
        }

        let formatted = match self.temp_celsius.is_nan() {
            true => format!("{}: NaN °C", Self::NAME),
            false => format!("{}: {:.2} °C", Self::NAME, self.temp_celsius),
        };
        let raw = format!("{:.2}", self.temp_celsius);

        MetricEntry::new(Self::NAME.to_string(), formatted, raw)
    }

    fn clear(&mut self) {}
}

impl Numeric for CpuTemperature {
    fn value(&self) -> f64 {
        self.temp_celsius as f64
    }
}
