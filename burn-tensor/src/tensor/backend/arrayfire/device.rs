#[derive(Copy, Clone, PartialEq, Debug)]
pub enum GPUBackend {
    CUDA,
    OPENCL,
}
pub type DeviceNumber = usize;

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Device {
    CPU,
    GPU(DeviceNumber, GPUBackend),
}
