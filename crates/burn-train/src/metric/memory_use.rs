/// RAM use metric
use super::{MetricMetadata, Numeric};
use crate::metric::{Metric, MetricEntry, NumericEntry};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use sysinfo::System;

/// Memory information
pub struct CpuMemory {
    name: Arc<String>,
    last_refresh: Instant,
    refresh_frequency: Duration,
    sys: System,
    ram_bytes_total: u64,
    ram_bytes_used: u64,
}

impl Clone for CpuMemory {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            last_refresh: self.last_refresh,
            refresh_frequency: self.refresh_frequency,
            sys: System::new(),
            ram_bytes_total: self.ram_bytes_total,
            ram_bytes_used: self.ram_bytes_used,
        }
    }
}

impl CpuMemory {
    /// Creates a new memory metric
    pub fn new() -> Self {
        let mut metric = Self {
            name: Arc::new("CPU Memory".into()),
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

        MetricEntry::new(self.name(), formatted, raw.to_string())
    }

    fn clear(&mut self) {}

    fn name(&self) -> Arc<String> {
        self.name.clone()
    }
}

impl Numeric for CpuMemory {
    fn value(&self) -> NumericEntry {
        NumericEntry::Value(bytes2gb(self.ram_bytes_used))
    }
}

fn bytes2gb(bytes: u64) -> f64 {
    bytes as f64 / 1e9
}
