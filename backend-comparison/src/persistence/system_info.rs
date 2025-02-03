use burn::serde::{Deserialize, Serialize};
use cubecl::wgpu::GraphicsApi;
use std::collections::HashSet;
use sysinfo;
use wgpu::{self, Backends};

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct BenchmarkSystemInfo {
    cpus: Vec<String>,
    gpus: Vec<String>,
    pub os: BenchmarkOSInfo,
}

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct BenchmarkOSInfo {
    pub name: String,
    #[serde(rename = "wsl")]
    windows_linux_subsystem: bool,
}

impl From<os_info::Info> for BenchmarkOSInfo {
    fn from(info: os_info::Info) -> Self {
        BenchmarkOSInfo {
            name: format!("{}", info),
            windows_linux_subsystem: wsl::is_wsl(),
        }
    }
}

impl BenchmarkSystemInfo {
    pub(crate) fn new() -> Self {
        Self {
            cpus: BenchmarkSystemInfo::enumerate_cpus(),
            gpus: BenchmarkSystemInfo::enumerate_gpus(),
            os: BenchmarkOSInfo::from(os_info::get()),
        }
    }

    fn enumerate_cpus() -> Vec<String> {
        let system = sysinfo::System::new_with_specifics(
            sysinfo::RefreshKind::nothing().with_cpu(sysinfo::CpuRefreshKind::everything()),
        );
        let cpu_names: HashSet<String> = system
            .cpus()
            .iter()
            .map(|c| c.brand().to_string())
            .collect();
        cpu_names.into_iter().collect()
    }

    fn enumerate_gpus() -> Vec<String> {
        let instance = wgpu::Instance::default();
        let adapters: Vec<wgpu::Adapter> = instance
            .enumerate_adapters({
                let backend = cubecl::wgpu::AutoGraphicsApi::backend();
                Backends::from_bits(1 << backend as u32).unwrap()
            })
            .into_iter()
            .filter(|adapter| {
                let info = adapter.get_info();
                info.device_type == wgpu::DeviceType::DiscreteGpu
                    || info.device_type == wgpu::DeviceType::IntegratedGpu
            })
            .collect();
        let gpu_names: HashSet<String> = adapters
            .iter()
            .map(|adapter| {
                let info = adapter.get_info();
                info.name
            })
            .collect();
        gpu_names.into_iter().collect()
    }
}
