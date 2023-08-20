/// RAM use metric
use super::MetricMetadata;
use crate::metric::{Metric, MetricEntry};
use sysinfo::{System, SystemExt};

static NAME: &str = "MEMORY";

/// Memory information
pub struct MemoryUse {
    ram_bytes_total: f32,
    ram_bytes_used: f32,
    swap_bytes_total: f32,
    swap_bytes_used: f32,
}

impl MemoryUse {
    /// Creates a new memory metric
    pub fn new() -> Self {
        Self {
            ram_bytes_total: 0.,
            ram_bytes_used: 0.,
            swap_bytes_total: 0.,
            swap_bytes_used: 0.,
        }
    }
}

impl Default for MemoryUse {
    fn default() -> Self {
        MemoryUse::new()
    }
}

impl Metric for MemoryUse {
    type Input = ();

    fn update(&mut self, _item: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        let mut sys = System::new();
        let mut memory_use = MemoryUse::new();

        // refreshing memory info before gathering it
        sys.refresh_memory();

        // bytes of RAM available
        memory_use.ram_bytes_total = sys.total_memory() as f32;

        // bytes of RAM in use
        memory_use.ram_bytes_used = sys.used_memory() as f32;

        // bytes of swap available
        memory_use.swap_bytes_total = sys.total_swap() as f32;

        // bytes of swap in use
        memory_use.swap_bytes_total = sys.used_swap() as f32;

        let ram_use_percentage = (memory_use.ram_bytes_used / memory_use.ram_bytes_total) * 100.;
        let swap_use_percentage = (memory_use.swap_bytes_used / memory_use.swap_bytes_total) * 100.;

        let formatted = format!(
            "RAM Used: {:.2}% - Swap Used: {:.2}%",
            ram_use_percentage, swap_use_percentage
        );

        let raw = format!(
            "ram: {:.2}%, swap: {:.2}%",
            ram_use_percentage, swap_use_percentage
        );

        MetricEntry::new(NAME.to_string(), formatted, raw)
    }

    fn clear(&mut self) {}
}
