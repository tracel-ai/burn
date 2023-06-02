/// The device struct when using the `wgpu` backend.
///
/// Note that you need to provide the device index when using a GPU backend.
///
/// # Example
///
/// ```no_run
/// use burn_wgpu::WGPUDevice;
///
/// let device_gpu_1 = WGPUDevice::DiscreteGPU(0); // First discrete GPU found.
/// let device_gpu_2 = WGPUDevice::DiscreteGPU(1);  // Second discrete GPU found.
/// ```
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum WGPUDevice {
    DiscreteGPU(usize),
    IntegratedGPU(usize),
    VirtualGPU(usize),
    CPU,
}

impl Default for WGPUDevice {
    fn default() -> Self {
        Self::CPU
    }
}
