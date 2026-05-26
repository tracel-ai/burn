#![recursion_limit = "256"]

use burn::tensor::{Device, DeviceConfig, Element};
use text_classification::AgNewsDataset;

#[cfg(not(feature = "f16"))]
#[allow(dead_code)]
type ElemType = f32;
#[cfg(feature = "f16")]
type ElemType = burn::tensor::f16;

pub fn launch(mut device: Device) {
    device
        .configure(DeviceConfig::default().float_dtype(ElemType::dtype()))
        .unwrap();

    text_classification::inference::infer::<AgNewsDataset>(
        device,
        "/tmp/text-classification-ag-news",
        // Samples from the test dataset, but you are free to test with your own text.
        vec![
            "Jays power up to take finale Contrary to popular belief, the power never really \
             snapped back at SkyDome on Sunday. The lights came on after an hour delay, but it \
             took some extra time for the batting orders to provide some extra wattage."
                .to_string(),
            "Yemen Sentences 15 Militants on Terror Charges A court in Yemen has sentenced one \
             man to death and 14 others to prison terms for a series of attacks and terrorist \
             plots in 2002, including the bombing of a French oil tanker."
                .to_string(),
            "IBM puts grids to work at U.S. Open IBM will put a collection of its On \
             Demand-related products and technologies to this test next week at the U.S. Open \
             tennis championships, implementing a grid-based infrastructure capable of running \
             multiple workloads including two not associated with the tournament."
                .to_string(),
        ],
    );
}

#[cfg(feature = "flex")]
mod flex {
    use burn::tensor::Device;

    pub fn run() {
        crate::launch(Device::flex());
    }
}

#[cfg(feature = "tch-gpu")]
mod tch_gpu {
    use burn::tensor::{Device, DeviceIndex};

    pub fn run() {
        #[cfg(not(target_os = "macos"))]
        let device = Device::libtorch_cuda(DeviceIndex::Default);
        #[cfg(target_os = "macos")]
        let device = Device::libtorch_mps();

        crate::launch(device);
    }
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use burn::tensor::Device;

    pub fn run() {
        crate::launch(Device::libtorch());
    }
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use burn::tensor::{Device, DeviceKind};

    pub fn run() {
        crate::launch(Device::wgpu(DeviceKind::DefaultDevice));
    }
}

#[cfg(feature = "metal")]
mod metal {
    use burn::tensor::{Device, DeviceKind};

    pub fn run() {
        crate::launch(Device::wgpu(DeviceKind::DefaultDevice));
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use burn::tensor::{Device, DeviceIndex};

    pub fn run() {
        crate::launch(Device::cuda(DeviceIndex::Default));
    }
}

fn main() {
    #[cfg(feature = "flex")]
    flex::run();
    #[cfg(feature = "tch-gpu")]
    tch_gpu::run();
    #[cfg(feature = "tch-cpu")]
    tch_cpu::run();
    #[cfg(feature = "wgpu")]
    wgpu::run();
    #[cfg(feature = "cuda")]
    cuda::run();
}
