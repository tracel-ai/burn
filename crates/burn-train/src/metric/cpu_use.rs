use super::{MetricMetadata, Numeric};
use crate::metric::{Metric, MetricEntry, NumericEntry};
use std::time::{Duration, Instant};
use sysinfo::{CpuRefreshKind, RefreshKind, System};

/// General CPU Usage metric
pub struct CpuUse {
    last_refresh: Instant,
    refresh_frequency: Duration,
    sys: System,
    current: f64,
}

impl CpuUse {
    /// Creates a new CPU metric
    pub fn new() -> Self {
        let mut sys = System::new();
        let current = Self::refresh(&mut sys);

        Self {
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

    fn name(&self) -> String {
        "CPU Usage".to_string()
    }
}

impl Numeric for CpuUse {
    fn value(&self) -> NumericEntry {
        NumericEntry::Value(self.current)
    }
}
