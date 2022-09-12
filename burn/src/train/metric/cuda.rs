use super::RunningMetricResult;
use crate::train::metric::{Metric, MetricState};
use nvml_wrapper::Nvml;

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

impl<T> Metric<T> for CUDAMetric {
    fn update(&mut self, _item: &T) -> Box<dyn MetricState> {
        let name = String::from("Cuda");

        let mut formatted = String::new();
        let mut raw_running = String::new();

        for index in 0..self.nvml.device_count().unwrap() {
            let device = self.nvml.device_by_index(index).unwrap();
            let memory_info = device.memory_info().unwrap();
            let used_gb = memory_info.used as f64 * 1e-9;
            let total_gb = memory_info.total as f64 * 1e-9;

            let memory_info_formatted = format!("{:.2}/{:.2} Gb", used_gb, total_gb);
            let memory_info_raw = format!("{}/{}", used_gb, total_gb);

            formatted = format!(
                "{} GPU #{} - Memory {}",
                formatted, index, memory_info_formatted
            );
            raw_running = format!("{} ", memory_info_raw);

            let utilization_rates = device.utilization_rates().unwrap();
            let utilization_rate_formatted = format!("{}%", utilization_rates.gpu);
            formatted = format!("{} - Usage {}", formatted, utilization_rate_formatted);
        }

        Box::new(RunningMetricResult {
            name,
            formatted,
            raw_running,
            raw_current: String::new(),
        })
    }

    fn clear(&mut self) {}
}
