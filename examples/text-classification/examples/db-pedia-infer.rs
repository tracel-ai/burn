use text_classification::DbPediaDataset;

use burn::tensor::backend::ADBackend;

#[cfg(not(feature = "f16"))]
#[allow(dead_code)]
type ElemType = f32;
#[cfg(feature = "f16")]
type ElemType = burn::tensor::f16;

pub fn launch<B: ADBackend>(device: B::Device) {
    text_classification::inference::infer::<B, DbPediaDataset>(
        device,
        "/tmp/text-classification-db-pedia",
        // Samples from the test dataset, but you are free to test with your own text.
        vec![
            " Magnus Eriksson is a Swedish former footballer who played as a forward.".to_string(),
            "Crossbeam Systems is headquartered in Boxborough Massachusetts and has offices in Europe Latin America and Asia Pacific. Crossbeam Systems was acquired by Blue Coat Systems in December 2012 and the Crossbeam brand has been fully absorbed into Blue Coat.".to_string(),
            " Zia is the sequel to the award-winning Island of the Blue Dolphins by Scott O'Dell. It was published in 1976 sixteen years after the publication of the first novel.".to_string(),
        ],
    );
}

#[cfg(any(
    feature = "ndarray",
    feature = "ndarray-blas-netlib",
    feature = "ndarray-blas-openblas",
    feature = "ndarray-blas-accelerate",
))]
mod ndarray {
    use burn_autodiff::ADBackendDecorator;
    use burn_ndarray::{NdArrayBackend, NdArrayDevice};

    use crate::{launch, ElemType};

    pub fn run() {
        launch::<ADBackendDecorator<NdArrayBackend<ElemType>>>(NdArrayDevice::Cpu);
    }
}

#[cfg(feature = "tch-gpu")]
mod tch_gpu {
    use burn_autodiff::ADBackendDecorator;
    use burn_tch::{TchBackend, TchDevice};

    use crate::{launch, ElemType};

    pub fn run() {
        #[cfg(not(target_os = "macos"))]
        let device = TchDevice::Cuda(0);
        #[cfg(target_os = "macos")]
        let device = TchDevice::Mps;

        launch::<ADBackendDecorator<TchBackend<ElemType>>>(device);
    }
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use burn_autodiff::ADBackendDecorator;
    use burn_tch::{TchBackend, TchDevice};

    use crate::{launch, ElemType};

    pub fn run() {
        launch::<ADBackendDecorator<TchBackend<ElemType>>>(TchDevice::Cpu);
    }
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use burn_autodiff::ADBackendDecorator;
    use burn_wgpu::{AutoGraphicsApi, WgpuBackend, WgpuDevice};

    use crate::{launch, ElemType};

    pub fn run() {
        launch::<ADBackendDecorator<WgpuBackend<AutoGraphicsApi, ElemType, i32>>>(
            WgpuDevice::default(),
        );
    }
}

fn main() {
    #[cfg(any(
        feature = "ndarray",
        feature = "ndarray-blas-netlib",
        feature = "ndarray-blas-openblas",
        feature = "ndarray-blas-accelerate",
    ))]
    ndarray::run();
    #[cfg(feature = "tch-gpu")]
    tch_gpu::run();
    #[cfg(feature = "tch-cpu")]
    tch_cpu::run();
    #[cfg(feature = "wgpu")]
    wgpu::run();
}
