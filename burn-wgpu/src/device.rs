/// The device struct when using the `wgpu` backend.
///
/// Note that you need to provide the device index when using a GPU backend.
///
/// # Example
///
/// ```no_run
/// use burn_wgpu::WgpuDevice;
///
/// let device_gpu_1 = WgpuDevice::DiscreteGpu(0); // First discrete GPU found.
/// let device_gpu_2 = WgpuDevice::DiscreteGpu(1);  // Second discrete GPU found.
/// ```
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum WgpuDevice {
    DiscreteGpu(usize),
    IntegratedGpu(usize),
    VirtualGpu(usize),
    Cpu,
}

impl Default for WgpuDevice {
    fn default() -> Self {
        #[cfg(target_os = "macos")]
        return Self::IntegratedGpu(0);

        #[cfg(not(target_os = "macos"))]
        Self::Cpu
    }
}
