#[cfg(any(
    feature = "ndarray",
    feature = "ndarray-blas-netlib",
    feature = "ndarray-blas-openblas",
    feature = "ndarray-blas-accelerate",
))]
mod ndarray {
    use burn::{
        backend::{
            ndarray::{NdArray, NdArrayDevice},
            Autodiff,
        },
        grad_clipping::GradientClippingConfig,
        optim::AdamConfig,
    };
    use lstm::{
        cli::{Cli, Commands},
        inference::infer,
        model::LstmNetworkConfig,
        training::{train, TrainingConfig},
    };

    pub fn run(cli: Cli) {
        let device = NdArrayDevice::Cpu;

        match cli.command {
            Commands::Train {
                artifact_dir,
                num_epochs,
                batch_size,
                num_workers,
                lr,
            } => {
                let config = TrainingConfig::new(
                    LstmNetworkConfig::new(),
                    // Gradient clipping via optimizer config
                    AdamConfig::new().with_grad_clipping(Some(GradientClippingConfig::Norm(1.0))),
                    num_epochs,
                    batch_size,
                    num_workers,
                    lr,
                );
                train::<Autodiff<NdArray>>(&artifact_dir, config, device);
            }
            Commands::Infer { artifact_dir } => {
                infer::<NdArray>(&artifact_dir, device);
            }
        }
    }
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use burn::{
        backend::{
            libtorch::{LibTorch, LibTorchDevice},
            Autodiff,
        },
        grad_clipping::GradientClippingConfig,
        optim::AdamConfig,
    };
    use lstm::{
        cli::{Cli, Commands},
        inference::infer,
        model::LstmNetworkConfig,
        training::{train, TrainingConfig},
    };

    pub fn run(cli: Cli) {
        let device = LibTorchDevice::Cpu;

        match cli.command {
            Commands::Train {
                artifact_dir,
                num_epochs,
                batch_size,
                num_workers,
                lr,
            } => {
                let config = TrainingConfig::new(
                    LstmNetworkConfig::new(),
                    // Gradient clipping via optimizer config
                    AdamConfig::new().with_grad_clipping(Some(GradientClippingConfig::Norm(1.0))),
                    num_epochs,
                    batch_size,
                    num_workers,
                    lr,
                );
                train::<Autodiff<LibTorch>>(&artifact_dir, config, device);
            }
            Commands::Infer { artifact_dir } => {
                infer::<LibTorch>(&artifact_dir, device);
            }
        }
    }
}

#[cfg(feature = "tch-gpu")]
mod tch_gpu {
    use burn::{
        backend::{
            libtorch::{LibTorch, LibTorchDevice},
            Autodiff,
        },
        grad_clipping::GradientClippingConfig,
        optim::AdamConfig,
    };
    use lstm::{
        cli::{Cli, Commands},
        inference::infer,
        model::LstmNetworkConfig,
        training::{train, TrainingConfig},
    };

    pub fn run(cli: Cli) {
        #[cfg(not(target_os = "macos"))]
        let device = LibTorchDevice::Cuda(0);
        #[cfg(target_os = "macos")]
        let device = LibTorchDevice::Mps;

        match cli.command {
            Commands::Train {
                artifact_dir,
                num_epochs,
                batch_size,
                num_workers,
                lr,
            } => {
                let config = TrainingConfig::new(
                    LstmNetworkConfig::new(),
                    // Gradient clipping via optimizer config
                    AdamConfig::new().with_grad_clipping(Some(GradientClippingConfig::Norm(1.0))),
                    num_epochs,
                    batch_size,
                    num_workers,
                    lr,
                );
                train::<Autodiff<LibTorch>>(&artifact_dir, config, device);
            }
            Commands::Infer { artifact_dir } => {
                infer::<LibTorch>(&artifact_dir, device);
            }
        }
    }
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use burn::{
        backend::{
            wgpu::{Wgpu, WgpuDevice},
            Autodiff,
        },
        grad_clipping::GradientClippingConfig,
        optim::AdamConfig,
    };
    use lstm::{
        cli::{Cli, Commands},
        inference::infer,
        model::LstmNetworkConfig,
        training::{train, TrainingConfig},
    };

    pub fn run(cli: Cli) {
        let device = WgpuDevice::default();

        match cli.command {
            Commands::Train {
                artifact_dir,
                num_epochs,
                batch_size,
                num_workers,
                lr,
            } => {
                let config = TrainingConfig::new(
                    LstmNetworkConfig::new(),
                    // Gradient clipping via optimizer config
                    AdamConfig::new().with_grad_clipping(Some(GradientClippingConfig::Norm(1.0))),
                    num_epochs,
                    batch_size,
                    num_workers,
                    lr,
                );
                train::<Autodiff<Wgpu>>(&artifact_dir, config, device);
            }
            Commands::Infer { artifact_dir } => {
                infer::<Wgpu>(&artifact_dir, device);
            }
        }
    }
}

use clap::Parser;
use lstm::cli::Cli;
fn main() {
    let cli = Cli::parse();
    #[cfg(any(
        feature = "ndarray",
        feature = "ndarray-blas-netlib",
        feature = "ndarray-blas-openblas",
        feature = "ndarray-blas-accelerate",
    ))]
    ndarray::run(cli);
    #[cfg(feature = "tch-cpu")]
    tch_cpu::run(cli);
    #[cfg(feature = "tch-gpu")]
    tch_gpu::run(cli);
    #[cfg(feature = "wgpu")]
    wgpu::run(cli);
}
