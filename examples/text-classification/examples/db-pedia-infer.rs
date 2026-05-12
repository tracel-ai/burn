use text_classification::DbPediaDataset;

use burn::tensor::{DType, Device, Element};

#[cfg(not(feature = "f16"))]
#[allow(dead_code)]
type ElemType = f32;
#[cfg(feature = "f16")]
type ElemType = burn::tensor::f16;

pub fn launch(device: impl Into<Device>) {
    let mut device = device.into();
    device
        .set_default_dtypes(ElemType::dtype(), DType::I32)
        .unwrap();

    text_classification::inference::infer::<DbPediaDataset>(
        device,
        "/tmp/text-classification-db-pedia",
        // Samples from the test dataset, but you are free to test with your own text.
        vec![
            " Magnus Eriksson is a Swedish former footballer who played as a forward.".to_string(),
            "Crossbeam Systems is headquartered in Boxborough Massachusetts and has offices in \
             Europe Latin America and Asia Pacific. Crossbeam Systems was acquired by Blue Coat \
             Systems in December 2012 and the Crossbeam brand has been fully absorbed into Blue \
             Coat."
                .to_string(),
            " Zia is the sequel to the award-winning Island of the Blue Dolphins by Scott O'Dell. \
             It was published in 1976 sixteen years after the publication of the first novel."
                .to_string(),
        ],
    );
}

#[cfg(feature = "flex")]
mod flex {
    use burn::backend::flex::FlexDevice;

    pub fn run() {
        crate::launch(FlexDevice);
    }
}

#[cfg(feature = "tch-gpu")]
mod tch_gpu {
    use burn::backend::libtorch::LibTorchDevice;

    pub fn run() {
        #[cfg(not(target_os = "macos"))]
        let device = LibTorchDevice::Cuda(0);
        #[cfg(target_os = "macos")]
        let device = LibTorchDevice::Mps;

        crate::launch(device);
    }
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use burn::backend::libtorch::LibTorchDevice;

    pub fn run() {
        crate::launch(LibTorchDevice::Cpu);
    }
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use burn::backend::wgpu::WgpuDevice;

    pub fn run() {
        crate::launch(WgpuDevice::default());
    }
}

#[cfg(feature = "metal")]
mod metal {
    use burn::backend::wgpu::WgpuDevice;

    pub fn run() {
        crate::launch(WgpuDevice::default());
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use burn::backend::cuda::CudaDevice;

    pub fn run() {
        crate::launch(CudaDevice::default());
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
