use std::sync::Arc;

use super::MetricMetadata;
use crate::metric::{Metric, MetricEntry, MetricName};
use nvml_wrapper::Nvml;

/// Track basic cuda infos.
#[derive(Clone)]
pub struct CudaMetric {
    name: MetricName,
    nvml: Arc<Option<Nvml>>,
}

impl CudaMetric {
    /// Creates a new metric for CUDA.
    pub fn new() -> Self {
        Self {
            name: Arc::new("Cuda".to_string()),
            nvml: Arc::new(Nvml::init().map(Some).unwrap_or_else(|err| {
                log::warn!("Unable to initialize CUDA Metric: {err}");
                None
            })),
        }
    }
}

impl Default for CudaMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for CudaMetric {
    type Input = ();

    fn update(&mut self, _item: &(), _metadata: &MetricMetadata) -> MetricEntry {
        let not_available = || {
            MetricEntry::new(
                self.name(),
                "Unavailable".to_string(),
                "Unavailable".to_string(),
            )
        };

        let available = |nvml: &Nvml| {
            let mut formatted = String::new();
            let mut raw_running = String::new();

            let device_count = match nvml.device_count() {
                Ok(val) => val,
                Err(err) => {
                    log::warn!("Unable to get the number of cuda devices: {err}");
                    return not_available();
                }
            };

            for index in 0..device_count {
                let device = match nvml.device_by_index(index) {
                    Ok(val) => val,
                    Err(err) => {
                        log::warn!("Unable to get device {index}: {err}");
                        return not_available();
                    }
                };
                let memory_info = match device.memory_info() {
                    Ok(info) => info,
                    Err(err) => {
                        log::warn!("Unable to get memory info from device {index}: {err}");
                        return not_available();
                    }
                };

                let used_gb = memory_info.used as f64 * 1e-9;
                let total_gb = memory_info.total as f64 * 1e-9;

                let memory_info_formatted = format!("{used_gb:.2}/{total_gb:.2} Gb");
                let memory_info_raw = format!("{used_gb}/{total_gb}");

                formatted = format!("{formatted} GPU #{index} - Memory {memory_info_formatted}");
                raw_running = format!("{memory_info_raw} ");

                let utilization_rates = match device.utilization_rates() {
                    Ok(rate) => rate,
                    Err(err) => {
                        log::warn!("Unable to get utilization rates from device {index}: {err}");
                        return not_available();
                    }
                };
                let utilization_rate_formatted = format!("{}%", utilization_rates.gpu);
                formatted = format!("{formatted} - Usage {utilization_rate_formatted}");
            }

            MetricEntry::new(self.name(), formatted, raw_running)
        };

        match self.nvml.as_ref() {
            Some(nvml) => available(nvml),
            None => not_available(),
        }
    }

    fn clear(&mut self) {}

    fn name(&self) -> MetricName {
        self.name.clone()
    }
}
