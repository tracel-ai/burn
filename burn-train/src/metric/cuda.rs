use super::{Adaptor, MetricMetadata};
use crate::metric::{Metric, MetricEntry};
use nvml_wrapper::Nvml;

/// Track basic cuda infos.
pub struct CUDAMetric {
    nvml: Nvml,
}

impl CUDAMetric {
    pub fn new() -> Self {
        Self {
            nvml: Nvml::init().unwrap(),
        }
    }
}

impl Default for CUDAMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Adaptor<()> for T {
    fn adapt(&self) {}
}

impl Metric for CUDAMetric {
    type Input = ();

    fn update(&mut self, _item: &(), _metadata: &MetricMetadata) -> MetricEntry {
        let name = String::from("Cuda");

        let mut formatted = String::new();
        let mut raw_running = String::new();

        for index in 0..self.nvml.device_count().unwrap() {
            let device = self.nvml.device_by_index(index).unwrap();
            let memory_info = device.memory_info().unwrap();
            let used_gb = memory_info.used as f64 * 1e-9;
            let total_gb = memory_info.total as f64 * 1e-9;

            let memory_info_formatted = format!("{used_gb:.2}/{total_gb:.2} Gb");
            let memory_info_raw = format!("{used_gb}/{total_gb}");

            formatted = format!("{formatted} GPU #{index} - Memory {memory_info_formatted}");
            raw_running = format!("{memory_info_raw} ");

            let utilization_rates = device.utilization_rates().unwrap();
            let utilization_rate_formatted = format!("{}%", utilization_rates.gpu);
            formatted = format!("{formatted} - Usage {utilization_rate_formatted}");
        }

        MetricEntry::new(name, formatted, raw_running)
    }

    fn clear(&mut self) {}
}
