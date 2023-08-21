/// CPU Temperature metric
use super::MetricMetadata;
use crate::metric::{Metric, MetricEntry};
use systemstat::{Platform, System};

static NAME: &str = "CPU_TEMP";

/// CPU Temperature in celsius degrees
pub struct CpuTemp {
    temp_celsius: f32,
}

impl CpuTemp {
    /// Creates a new CPU temp metric
    pub fn new() -> Self {
        Self { temp_celsius: 0. }
    }
}

impl Default for CpuTemp {
    fn default() -> Self {
        CpuTemp::new()
    }
}

impl Metric for CpuTemp {
    type Input = ();

    fn update(&mut self, _item: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        let sys = System::new();

        match sys.cpu_temp() {
            Ok(temp) => self.temp_celsius = temp,
            Err(_) => self.temp_celsius = f32::NAN,
        }

        let formatted = format!("CPU Temp: {:.2}°C", self.temp_celsius);

        let raw = format!("{:.2}", self.temp_celsius);

        MetricEntry::new(NAME.to_string(), formatted, raw)
    }

    fn clear(&mut self) {}
}
