/// GPU Temperature metric

use super::MetricMetadata;
use crate::metric::{Metric, MetricEntry};
use sysinfo::{ComponentExt, System, SystemExt};

static NAME: &str = "GPU_TEMP";

/// GPU Temperature in celsius degrees
pub struct GpuTemp {
    temp_celsius: f32,
}

impl GpuTemp {
    /// Creates a new GPU temp metric
    pub fn new() -> Self {
        Self {
            temp_celsius: 0.
        }
    }
}

impl Default for GpuTemp {
    fn default() -> Self {
        GpuTemp::new()
    }
}

impl Metric for GpuTemp {
    type Input = ();

    fn update(&mut self, _item: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        let mut sys = System::new();

        let mut gpu_temp = GpuTemp::new();
        // Vec containing all "gpu" labeled devices' temperature
        let mut temps_vec: Vec<f32> = Vec::new();

        // refreshing components' info
        sys.refresh_components_list();
        sys.refresh_components();

        for component in sys.components().iter() {
            // if the component is a gpu, its temperature is added to the `temps` vec.
            // Then, the mean of all these temps will be calculated
            if component.label().to_lowercase().contains("gpu") {
                // saving the temperature
                temps_vec.push(component.temperature());
            }
        }

        gpu_temp.temp_celsius = temps_vec.iter().sum::<f32>() / temps_vec.len() as f32;

        let mut formatted = String::new();

        if temps_vec.len() > 1 {
            formatted = format!("Mean of GPUs temps: {:.2}°C", gpu_temp.temp_celsius)
        }
        else {
            formatted = format!("GPU Temp: {:.2}°C", gpu_temp.temp_celsius)
        }

        let raw = format!("{:.2}", gpu_temp.temp_celsius);

        MetricEntry::new(NAME.to_string(), formatted, raw)
    }

    fn clear(&mut self) {}
}
