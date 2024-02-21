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
    /// Discrete GPU with the given index. The index is the index of the discrete GPU in the list
    /// of all discrete GPUs found on the system.
    DiscreteGpu(usize),

    /// Integrated GPU with the given index. The index is the index of the integrated GPU in the
    /// list of all integrated GPUs found on the system.
    IntegratedGpu(usize),

    /// Virtual GPU with the given index. The index is the index of the virtual GPU in the list of
    /// all virtual GPUs found on the system.
    VirtualGpu(usize),

    /// CPU.
    Cpu,

    /// The best available device found with the current [graphics API](crate::GraphicsApi).
    ///
    /// Priority
    ///
    ///   1. DiscreteGpu
    ///   2. IntegratedGpu
    ///   3. VirtualGpu
    ///   4. Cpu
    ///
    /// # Notes
    ///
    /// A device might be identified as [Other](wgpu::DeviceType::Other) by [wgpu](wgpu), in this case, we chose this device over
    /// `IntegratedGpu` since it's often a discrete GPU.
    BestAvailable,
}

impl Default for WgpuDevice {
    fn default() -> Self {
        Self::BestAvailable
    }
}
