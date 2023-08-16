/// GPU Temperature metric

use super::MetricMetadata;
use crate::metric::{Metric, MetricEntry};
use sysinfo::{ComponentExt, System, SystemExt, Component};

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

        // refreshing components' info
        sys.refresh_components_list();
        sys.refresh_components();

        // vec containing components of the system (CPU, disks, NIC, GPU etc...)
        let components_list: Vec<&Component> = sys.components().iter().collect();

        for component in components_list {
            // if the component is a gpu, its temperature is saved to the struct
            if component.label().to_lowercase().contains("gpu") {
                // saving the temperature
                gpu_temp.temp_celsius = component.temperature();
            }
        }
        
        let formatted = format!("GPU Temp: {:.2}°C", gpu_temp.temp_celsius);

        let raw = format!("{:.2}", gpu_temp.temp_celsius);

        MetricEntry::new(NAME.to_string(), formatted, raw)
    }

    fn clear(&mut self) {}
}

