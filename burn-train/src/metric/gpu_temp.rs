use super::MetricMetadata;
use crate::metric::{Metric, MetricEntry};
use std::time::{Duration, Instant};
use sysinfo::{ComponentExt, System, SystemExt};

/// GPU Temperature in celsius degrees
pub struct GpuTemp {
    last_refresh: Instant,
    refresh_frequency: Duration,
    sys: System,
    current: Vec<f64>,
}

impl GpuTemp {
    /// Creates a new GPU temp metric
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

    fn refresh(sys: &mut System) -> Vec<f64> {
        // refreshing components' info
        sys.refresh_components_list();
        sys.refresh_components();

        let components = sys.components();
        let mut temps = Vec::new();

        for component in components {
            if component.label().to_lowercase().contains("gpu") {
                temps.push(component.temperature() as f64);
            }
        }

        temps
    }
}

impl Default for GpuTemp {
    fn default() -> Self {
        GpuTemp::new()
    }
}

impl Metric for GpuTemp {
    const NAME: &'static str = "GPU Temperature";

    type Input = ();

    fn update(&mut self, _item: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        if self.last_refresh.elapsed() >= self.refresh_frequency {
            self.current = Self::refresh(&mut self.sys);
            self.last_refresh = Instant::now();
        }

        let mean_temp = self.current.iter().sum::<f64>() / self.current.len() as f64;

        // if there is more than 1 GPU, the metric lets the user know that the value displayed is a mean
        let formatted = if self.current.len() > 1 {
            format!("Mean of GPUs temps: {:.2}°C", mean_temp)
        } else {
            format!("GPU Temp: {:.2}°C", mean_temp)
        };

        let raw = format!("{:.2}", mean_temp);

        MetricEntry::new(Self::NAME.to_string(), formatted, raw)
    }

    fn clear(&mut self) {}
}
