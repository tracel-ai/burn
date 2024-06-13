use text_classification::DbPediaDataset;

use burn::tensor::backend::AutodiffBackend;

#[cfg(not(feature = "f16"))]
#[allow(dead_code)]
type ElemType = f32;
#[cfg(feature = "f16")]
type ElemType = burn::tensor::f16;

pub fn launch<B: AutodiffBackend>(device: B::Device) {
    text_classification::inference::infer::<B, DbPediaDataset>(
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

#[cfg(any(
    feature = "ndarray",
    feature = "ndarray-blas-netlib",
    feature = "ndarray-blas-openblas",
    feature = "ndarray-blas-accelerate",
))]
mod ndarray {
    use burn::backend::{
        ndarray::{NdArray, NdArrayDevice},
        Autodiff,
    };

    use crate::{launch, ElemType};

    pub fn run() {
        launch::<Autodiff<NdArray<ElemType>>>(NdArrayDevice::Cpu);
    }
}

#[cfg(feature = "tch-gpu")]
mod tch_gpu {
    use burn::backend::{
        libtorch::{LibTorch, LibTorchDevice},
        Autodiff,
    };

    use crate::{launch, ElemType};

    pub fn run() {
        #[cfg(not(target_os = "macos"))]
        let device = LibTorchDevice::Cuda(0);
        #[cfg(target_os = "macos")]
        let device = LibTorchDevice::Mps;

        launch::<Autodiff<LibTorch<ElemType>>>(device);
    }
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use burn::backend::{
        tch::{LibTorch, LibTorchDevice},
        Autodiff,
    };

    use crate::{launch, ElemType};

    pub fn run() {
        launch::<Autodiff<LibTorch<ElemType>>>(LibTorchDevice::Cpu);
    }
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use burn::backend::{
        wgpu::{Wgpu, WgpuDevice},
        Autodiff,
    };

    use crate::{launch, ElemType};

    pub fn run() {
        launch::<Autodiff<Wgpu<ElemType, i32>>>(WgpuDevice::default());
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
