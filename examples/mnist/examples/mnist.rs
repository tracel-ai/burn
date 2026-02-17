#![recursion_limit = "256"]

use burn::{Device, Dispatch, backend::Autodiff};
use mnist::training;

#[cfg(feature = "cuda")]
use burn::backend::cuda::CudaDevice;
#[cfg(feature = "tch-gpu")]
use burn::backend::libtorch::LibTorchDevice;
#[cfg(feature = "ndarray")]
use burn::backend::ndarray::NdArrayDevice;
#[cfg(feature = "rocm")]
use burn::backend::rocm::RocmDevice;
#[cfg(any(feature = "wgpu", feature = "metal", feature = "vulkan"))]
use burn::backend::wgpu::WgpuDevice;

// TODO: refactor lib to use Dispatch

#[allow(unreachable_code)]
fn select_device() -> Device {
    #[cfg(feature = "ndarray")]
    return NdArrayDevice::Cpu.into();

    #[cfg(all(feature = "tch-gpu", not(target_os = "macos")))]
    LibTorchDevice::Cuda(0).into();

    #[cfg(all(feature = "tch-gpu", target_os = "macos"))]
    return LibTorchDevice::Mps.into();

    #[cfg(feature = "tch-cpu")]
    return LibTorchDevice::Cpu;

    #[cfg(any(feature = "wgpu", feature = "metal", feature = "vulkan"))]
    return WgpuDevice::default().into();

    #[cfg(feature = "cuda")]
    return CudaDevice::default().into();

    #[cfg(feature = "rocm")]
    return RocmDevice::default().into();

    unreachable!("At least one backend will be selected.")
}

fn main() {
    let device = select_device();
    training::run::<Autodiff<Dispatch>>(device);
}
