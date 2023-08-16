/// CPU Temperature metric

use super::MetricMetadata;
use crate::metric::{Metric, MetricEntry};
use sysinfo::{ComponentExt, System, SystemExt, Component};

static NAME: &str = "CPU_TEMP";

/// CPU Temperature in celsius degrees
pub struct CpuTemp {
    temp_celsius: f32,
}

impl CpuTemp {
    /// Creates a new CPU temp metric
    pub fn new() -> Self {
        Self {
            temp_celsius: 0.
        }
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
        let mut sys = System::new();

        let mut cpu_temp = CpuTemp::new();

        // refreshing components' info
        sys.refresh_components_list();
        sys.refresh_components();

        // vec containing components of the system (CPU, disks, NIC, GPU etc...)
        let components_list: Vec<&Component> = sys.components().iter().collect();

        for component in components_list {
            // acpi corresponds to the CPU. if we find acpi thermal zone or something like that,
            // we know it's the CPU
            if component.label().to_lowercase().contains("acpi") {
                // update CPU temp
                cpu_temp.temp_celsius = component.temperature();
            }
        }
        
        let formatted = format!("CPU Temp: {:.2}°C", cpu_temp.temp_celsius);

        let raw = format!("{:.2}", cpu_temp.temp_celsius);

        MetricEntry::new(NAME.to_string(), formatted, raw)
    }

    fn clear(&mut self) {}
}
