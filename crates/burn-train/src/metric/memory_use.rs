/// RAM use metric
use super::{MetricMetadata, Numeric};
use crate::metric::{Metric, MetricEntry};
use std::time::{Duration, Instant};
use sysinfo::System;

/// Memory information
pub struct CpuMemory {
    last_refresh: Instant,
    refresh_frequency: Duration,
    sys: System,
    ram_bytes_total: u64,
    ram_bytes_used: u64,
}

impl CpuMemory {
    /// Creates a new memory metric
    pub fn new() -> Self {
        let mut metric = Self {
            last_refresh: Instant::now(),
            refresh_frequency: Duration::from_millis(200),
            sys: System::new(),
            ram_bytes_total: 0,
            ram_bytes_used: 0,
        };
        metric.refresh();
        metric
    }

    fn refresh(&mut self) {
        self.sys.refresh_memory();
        self.last_refresh = Instant::now();

        // bytes of RAM available
        self.ram_bytes_total = self.sys.total_memory();

        // bytes of RAM in use
        self.ram_bytes_used = self.sys.used_memory();
    }
}

impl Default for CpuMemory {
    fn default() -> Self {
        CpuMemory::new()
    }
}

impl Metric for CpuMemory {
    const NAME: &'static str = "CPU Memory";

    type Input = ();

    fn update(&mut self, _item: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        if self.last_refresh.elapsed() >= self.refresh_frequency {
            self.refresh();
        }

        let raw = bytes2gb(self.ram_bytes_used);
        let formatted = format!(
            "RAM Used: {:.2} / {:.2} Gb",
            raw,
            bytes2gb(self.ram_bytes_total),
        );

        MetricEntry::new(Self::NAME.to_string(), formatted, raw.to_string())
    }

    fn clear(&mut self) {}
}

impl Numeric for CpuMemory {
    fn value(&self) -> f64 {
        bytes2gb(self.ram_bytes_used)
    }
}

fn bytes2gb(bytes: u64) -> f64 {
    bytes as f64 / 1e9
}
