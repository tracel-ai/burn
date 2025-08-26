use super::{MetricMetadata, Numeric};
use crate::metric::{Metric, MetricEntry, MetricName, NumericEntry};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use sysinfo::{CpuRefreshKind, RefreshKind, System};

/// General CPU Usage metric
pub struct CpuUse {
    name: MetricName,
    last_refresh: Instant,
    refresh_frequency: Duration,
    sys: System,
    current: f64,
}

impl Clone for CpuUse {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            last_refresh: self.last_refresh,
            refresh_frequency: self.refresh_frequency,
            sys: System::new(),
            current: self.current,
        }
    }
}

impl CpuUse {
    /// Creates a new CPU metric
    pub fn new() -> Self {
        let mut sys = System::new();
        let current = Self::refresh(&mut sys);
        let name = "CPU Usage".to_string();

        Self {
            name: Arc::new(name),
            last_refresh: Instant::now(),
            refresh_frequency: Duration::from_millis(200),
            sys,
            current,
        }
    }

    fn refresh(sys: &mut System) -> f64 {
        sys.refresh_specifics(
            RefreshKind::nothing().with_cpu(CpuRefreshKind::nothing().with_cpu_usage()),
        );

        let cpus = sys.cpus();
        let num_cpus = cpus.len();
        let use_percentage = cpus.iter().fold(0.0, |acc, cpu| acc + cpu.cpu_usage()) as f64;

        use_percentage / num_cpus as f64
    }
}

impl Default for CpuUse {
    fn default() -> Self {
        CpuUse::new()
    }
}

impl Metric for CpuUse {
    type Input = ();

    fn update(&mut self, _item: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        if self.last_refresh.elapsed() >= self.refresh_frequency {
            self.current = Self::refresh(&mut self.sys);
            self.last_refresh = Instant::now();
        }

        let formatted = format!("{}: {:.2} %", self.name(), self.current);
        let raw = format!("{:.2}", self.current);

        MetricEntry::new(self.name(), formatted, raw)
    }

    fn clear(&mut self) {}

    fn name(&self) -> MetricName {
        self.name.clone()
    }
}

impl Numeric for CpuUse {
    fn value(&self) -> NumericEntry {
        NumericEntry::Value(self.current)
    }
}
