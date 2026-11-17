//! CPU Temperature metric

use std::sync::Arc;

use super::MetricMetadata;
use crate::metric::{Metric, MetricAttributes, MetricName, Numeric, NumericEntry, SerializedEntry};
use burn_core::tensor::TensorReadError;
use systemstat::{Platform, System};

/// CPU Temperature in celsius degrees
#[derive(Clone)]
pub struct CpuTemperature {
    name: MetricName,
    temp_celsius: f32,
}

impl CpuTemperature {
    /// Creates a new CPU temp metric
    pub fn new() -> Self {
        let name = Arc::new("CPU Temperature".to_string());

        Self {
            name,
            temp_celsius: 0.,
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

    fn update(
        &mut self,
        _item: &Self::Input,
        _metadata: &MetricMetadata,
    ) -> Result<SerializedEntry, TensorReadError> {
        self.compute()
    }

    fn compute(&mut self) -> Result<SerializedEntry, TensorReadError> {
        // A new `System` for each reading: systemstat 0.2.8 holds a `RefCell` (not `Sync`) and a
        // bug prevents a reused `System` to read the sensor more than once.
        match System::new().cpu_temp() {
            Ok(temp) => self.temp_celsius = temp,
            Err(_) => self.temp_celsius = f32::NAN,
        }

        let formatted = match self.temp_celsius.is_nan() {
            true => format!("{}: NaN °C", self.name()),
            false => format!("{}: {:.2} °C", self.name(), self.temp_celsius),
        };
        let raw = format!("{:.2}", self.temp_celsius);

        Ok(SerializedEntry::new(formatted, raw))
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
    fn value(&self) -> Option<NumericEntry> {
        Some(NumericEntry::Value(self.temp_celsius as f64))
    }

    fn running_value(&self) -> Option<NumericEntry> {
        Some(NumericEntry::Value(self.temp_celsius as f64))
    }

    fn final_value(&self) -> NumericEntry {
        NumericEntry::Value(self.temp_celsius as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test: with systemstat 0.2.8 a reused `System` reads the
    /// sensor only once, so the second reading came back as NaN. On a machine
    /// without a readable sensor, both readings are NaN.
    #[test]
    fn a_second_reading_reads_the_sensor_like_the_first() {
        let mut metric = CpuTemperature::new();

        metric.compute().unwrap();
        let first = metric.temp_celsius;
        metric.compute().unwrap();
        let second = metric.temp_celsius;

        assert_eq!(first.is_nan(), second.is_nan());
    }
}
