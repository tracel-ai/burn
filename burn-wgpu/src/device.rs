#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum WGPUDevice {
    DiscreteGPU(usize),
    IntegratedGPU(usize),
    VirtualGPU(usize),
    CPU,
}

impl Default for WGPUDevice {
    fn default() -> Self {
        Self::DiscreteGPU(0)
    }
}
