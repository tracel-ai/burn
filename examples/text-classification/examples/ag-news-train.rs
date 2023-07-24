use burn::nn::transformer::TransformerEncoderConfig;
use burn::optim::{decay::WeightDecayConfig, AdamConfig};
use burn::tensor::backend::{ADBackend, Backend};

use burn_autodiff::ADBackendDecorator;
use text_classification::training::ExperimentConfig;
use text_classification::AgNewsDataset;

#[cfg(not(feature = "f16"))]
type ElemType = f32;
#[cfg(feature = "f16")]
type ElemType = burn::tensor::f16;

// #[cfg(feature = "tch-cpu")]
// type Backend = TchBackend<ElemType>;
// #[cfg(feature = "tch-gpu")]
// type Backend = TchBackend<ElemType>;
// #[cfg(feature = "wgpu")]
// type Backend = WgpuBackend;

// type ADBackend = burn_autodiff::ADBackendDecorator<Backend>;

// fn main() {
//     let config = ExperimentConfig::new(
//         TransformerEncoderConfig::new(256, 1024, 8, 4).with_norm_first(true),
//         AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5))),
//     );

//     text_classification::training::train::<ADBackend, AgNewsDataset>(
//         if cfg!(target_os = "macos") {
//             TchDevice::Mps
//         } else {
//             TchDevice::Cuda(0)
//         },
//         AgNewsDataset::train(),
//         AgNewsDataset::test(),
//         config,
//         "/tmp/text-classification-ag-news",
//     );
// }

pub fn launch<B: ADBackend>(device: B::Device) {
    let config = ExperimentConfig::new(
        TransformerEncoderConfig::new(256, 1024, 8, 4).with_norm_first(true),
        AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5))),
    );

    text_classification::training::train::<B, AgNewsDataset>(
        device,
        AgNewsDataset::train(),
        AgNewsDataset::test(),
        config,
        "/tmp/text-classification-ag-news",
    );
}

#[cfg(any(
    feature = "ndarray",
    feature = "ndarray-blas-netlib",
    feature = "ndarray-blas-openblas",
    feature = "ndarray-blas-accelerate",
))]
mod ndarray {
    use burn_ndarray::{NdArrayBackend, NdArrayDevice};

    use crate::run;
    type ADBackend = burn_autodiff::ADBackendDecorator<NdArrayBackend<f32>>;

    pub fn run() {
        launch::<ADBackend>(NdArrayDevice::Cpu);
    }
}

#[cfg(feature = "tch-gpu")]
mod tch_gpu {
    use burn_autodiff::ADBackendDecorator;
    use burn_tch::{TchBackend, TchDevice};
    use mnist::training;

    pub fn run() {
        #[cfg(not(target_os = "macos"))]
        let device = TchDevice::Cuda(0);
        #[cfg(target_os = "macos")]
        let device = TchDevice::Mps;

        training::run::<ADBackendDecorator<TchBackend<f32>>>(device);
    }
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use burn_autodiff::ADBackendDecorator;
    use burn_wgpu::{AutoGraphicsApi, WgpuBackend, WgpuDevice};
    use mnist::training;

    pub fn run() {
        let device = WgpuDevice::default();
        training::run::<ADBackendDecorator<WgpuBackend<AutoGraphicsApi, f32, i32>>>(device);
    }
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use burn_autodiff::ADBackendDecorator;
    use burn_tch::{TchBackend, TchDevice};
    use mnist::training;

    pub fn run() {
        let device = TchDevice::Cpu;
        training::run::<ADBackendDecorator<TchBackend<f32>>>(device);
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
