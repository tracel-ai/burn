/// The CPU use metric.
use super::MetricMetadata;
use crate::metric::{Metric, MetricEntry};
use sysinfo::{CpuExt, System, SystemExt};

static NAME: &str = "CPU_USE";

/// General CPU Usage metric
pub struct CpuUse {
    use_percentage: f32,
}

impl CpuUse {
    /// Creates a new CPU metric
    pub fn new() -> Self {
        Self { use_percentage: 0. }
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
        let mut sys = System::new();

        // variables are declared here so that we can still access them after the 'for' loop
        let mut formatted = String::new();
        let mut raw = String::new();

        // CPU data is gathered twice because all values are 0 the first time
        // sysinfo documentation says the following:
        // "Please note that the result will very likely be inaccurate at the first call.
        // You need to call this method at least twice with a bit of time between each call" (see line 59)
        for _i in 0..=1 {
            let mut cores_use: Vec<f32> = Vec::new();

            sys.refresh_cpu(); // Refreshing CPU information

            for cpu in sys.cpus() {
                // use percentage of each core
                cores_use.push(cpu.cpu_usage());
            }
            // Mean of all cores use -> General CPU use
            self.use_percentage = cores_use.iter().sum::<f32>() / sys.cpus().len() as f32;
            formatted = format!("CPU Use: {:.2}%", self.use_percentage);
            raw = format!("{:.2}", self.use_percentage);

            std::thread::sleep(System::MINIMUM_CPU_UPDATE_INTERVAL);
        }

        MetricEntry::new(NAME.to_string(), formatted, raw)
    }

    fn clear(&mut self) {}
}
