#![recursion_limit = "256"]

//! Benchmarks for PyTorch model loading performance.
//!
//! This benchmark compares loading PyTorch models (.pth files) using:
//! 1. The old PyTorchFileRecorder from burn-import
//! 2. The new PytorchStore from burn-store
//!
//! The benchmark tests a single model size (configurable via environment variable)
//! across different backends and loading methods.
//!
//! Before running these benchmarks, generate the PyTorch model files:
//! ```bash
//! cd crates/burn-store
//! uv run benches/generate_pytorch_models.py
//! ```
//!
//! Then run the benchmarks:
//! ```bash
//! # Run with default (small) model
//! cargo bench --bench pytorch_loading
//!
//! # Run with specific model size
//! MODEL_SIZE=medium cargo bench --bench pytorch_loading
//! MODEL_SIZE=large cargo bench --bench pytorch_loading
//! ```
//!
//! Run with specific backends:
//! ```bash
//! cargo bench --bench pytorch_loading --features metal
//! cargo bench --bench pytorch_loading --features cuda
//! cargo bench --bench pytorch_loading --features wgpu
//! ```

use burn_core::module::Module;
use burn_core::nn;
use burn_core::prelude::*;
use burn_core::record::{FullPrecisionSettings, Recorder};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};
use burn_store::{ModuleSnapshot, PytorchStore};
use divan::{AllocProfiler, Bencher};
use std::fs;
use std::path::PathBuf;

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

// Backend type aliases for easy switching
type NdArrayBackend = burn_ndarray::NdArray<f32>;

#[cfg(feature = "wgpu")]
type WgpuBackend = burn_wgpu::Wgpu;

#[cfg(feature = "cuda")]
type CudaBackend = burn_cuda::Cuda<f32, i32>;

#[cfg(feature = "candle")]
type CandleBackend = burn_candle::Candle<f32, i64>;

#[cfg(feature = "tch")]
type TchBackend = burn_tch::LibTorch<f32>;

#[cfg(feature = "metal")]
type MetalBackend = burn_metal::Metal;

// Model size enum
#[derive(Debug, Clone, Copy)]
enum ModelSize {
    Small,
    Medium,
    Large,
}

impl ModelSize {
    fn all() -> &'static [Self] {
        &[Self::Small, Self::Medium, Self::Large]
    }
}

// Simple model for testing (small)
#[derive(Module, Debug)]
struct SimpleModel<B: Backend> {
    fc1: nn::Linear<B>,
    fc2: nn::Linear<B>,
}

impl<B: Backend> SimpleModel<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            fc1: nn::LinearConfig::new(784, 128).init(device),
            fc2: nn::LinearConfig::new(128, 10).init(device),
        }
    }
}

// Medium model to test typical workloads
#[derive(Module, Debug)]
struct MediumModel<B: Backend> {
    conv1: nn::conv::Conv2d<B>,
    conv2: nn::conv::Conv2d<B>,
    fc1: nn::Linear<B>,
    fc2: nn::Linear<B>,
}

impl<B: Backend> MediumModel<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            conv1: nn::conv::Conv2dConfig::new([3, 32], [3, 3]).init(device),
            conv2: nn::conv::Conv2dConfig::new([32, 64], [3, 3]).init(device),
            fc1: nn::LinearConfig::new(1024, 256).init(device),
            fc2: nn::LinearConfig::new(256, 10).init(device),
        }
    }
}

// Large model to test scalability
#[derive(Module, Debug)]
struct LargeModel<B: Backend> {
    layers: Vec<nn::Linear<B>>,
}

impl<B: Backend> LargeModel<B> {
    fn new(device: &B::Device) -> Self {
        let mut layers = Vec::new();
        // Create a model with 20 layers
        for i in 0..20 {
            let in_size = if i == 0 { 1024 } else { 2048 };
            layers.push(nn::LinearConfig::new(in_size, 2048).init(device));
        }
        Self { layers }
    }
}

/// Get the path to the PyTorch model files
fn get_model_dir() -> PathBuf {
    // Try common temp directories
    let possible_dirs = [
        PathBuf::from("/tmp/pytorch_bench_models"),
        PathBuf::from("/var/folders").join("pytorch_bench_models"),
        std::env::temp_dir().join("pytorch_bench_models"),
    ];

    for dir in &possible_dirs {
        if dir.exists() {
            return dir.clone();
        }
    }

    // If none exist, use the first one (will fail with clear error)
    possible_dirs[0].clone()
}

/// Get model file paths based on size
fn get_model_paths(size: ModelSize) -> (PathBuf, PathBuf, PathBuf) {
    let model_dir = get_model_dir();
    let prefix = match size {
        ModelSize::Small => "simple_model",
        ModelSize::Medium => "medium_model",
        ModelSize::Large => "large_model",
    };

    (
        model_dir.join(format!("{}_state_dict.pth", prefix)),
        model_dir.join(format!("{}_checkpoint.pth", prefix)),
        model_dir.join(format!("{}_wrapped.pth", prefix)),
    )
}

/// Check if model files exist and provide helpful error message
fn check_model_files() -> Result<PathBuf, String> {
    let model_dir = get_model_dir();

    if !model_dir.exists() {
        return Err(format!(
            "\n❌ PyTorch model files not found!\n\
            \n\
            Please generate the model files first by running:\n\
            \n\
            cd crates/burn-store\n\
            uv run benches/generate_pytorch_models.py\n\
            \n\
            Expected directory: {}\n",
            model_dir.display()
        ));
    }

    // Check if at least one model file exists
    let test_file = model_dir.join("simple_model_state_dict.pth");
    if !test_file.exists() {
        return Err(format!(
            "\n❌ Model directory exists but files are missing!\n\
            \n\
            Please regenerate the model files by running:\n\
            \n\
            cd crates/burn-store\n\
            uv run benches/generate_pytorch_models.py\n\
            \n\
            Looking for: {}\n",
            test_file.display()
        ));
    }

    Ok(model_dir)
}

fn main() {
    // Check if model files exist before running benchmarks
    match check_model_files() {
        Ok(dir) => {
            println!("✅ Found PyTorch model files in: {}", dir.display());
            println!("📊 Benchmarking all model sizes: small, medium, large");
            println!("\n🚀 Running PyTorch loading benchmarks...\n");

            println!("Available backends:");
            println!("  - NdArray (CPU)");
            #[cfg(feature = "wgpu")]
            println!("  - WGPU (GPU)");
            #[cfg(feature = "cuda")]
            println!("  - CUDA (NVIDIA GPU)");
            #[cfg(feature = "candle")]
            println!("  - Candle");
            #[cfg(feature = "tch")]
            println!("  - LibTorch");
            #[cfg(feature = "metal")]
            println!("  - Metal (Apple GPU)");
            println!();

            divan::main();
        }
        Err(msg) => {
            eprintln!("{}", msg);
            std::process::exit(1);
        }
    }
}

// Macro to generate benchmarks for each backend
macro_rules! bench_backend {
    ($backend:ty, $mod_name:ident, $backend_name:literal) => {
        #[divan::bench_group(name = $backend_name)]
        mod $mod_name {
            use super::*;

            type TestBackend = $backend;
            type TestDevice = <TestBackend as Backend>::Device;

            #[divan::bench(sample_count = 20, args = ModelSize::all())]
            fn old_recorder(bencher: Bencher, model_size: &ModelSize) {
                let (_, checkpoint_path, _) = get_model_paths(*model_size);

                // Skip if file doesn't exist
                if !checkpoint_path.exists() {
                    eprintln!("Skipping: {} not found", checkpoint_path.display());
                    return;
                }

                let file_size = fs::metadata(&checkpoint_path).unwrap().len();

                bencher
                    .counter(divan::counter::BytesCount::new(file_size))
                    .bench(|| {
                        let device: TestDevice = Default::default();
                        let recorder = PyTorchFileRecorder::<FullPrecisionSettings>::default();
                        let load_args = LoadArgs::new(checkpoint_path.clone())
                            .with_top_level_key("model_state_dict");

                        match model_size {
                            ModelSize::Small => {
                                let record =
                                    recorder.load(load_args, &device).expect("Failed to load");
                                let _model =
                                    SimpleModel::<TestBackend>::new(&device).load_record(record);
                            }
                            ModelSize::Medium => {
                                let record =
                                    recorder.load(load_args, &device).expect("Failed to load");
                                let _model =
                                    MediumModel::<TestBackend>::new(&device).load_record(record);
                            }
                            ModelSize::Large => {
                                let record =
                                    recorder.load(load_args, &device).expect("Failed to load");
                                let _model =
                                    LargeModel::<TestBackend>::new(&device).load_record(record);
                            }
                        }
                    });
            }

            #[divan::bench(sample_count = 20, args = ModelSize::all())]
            fn new_store(bencher: Bencher, model_size: &ModelSize) {
                let (_, checkpoint_path, _) = get_model_paths(*model_size);

                // Skip if file doesn't exist
                if !checkpoint_path.exists() {
                    eprintln!("Skipping: {} not found", checkpoint_path.display());
                    return;
                }

                let file_size = fs::metadata(&checkpoint_path).unwrap().len();

                bencher
                    .counter(divan::counter::BytesCount::new(file_size))
                    .bench(|| {
                        let device: TestDevice = Default::default();

                        match model_size {
                            ModelSize::Small => {
                                let mut model = SimpleModel::<TestBackend>::new(&device);
                                let mut store = PytorchStore::from_file(checkpoint_path.clone())
                                    .with_top_level_key("model_state_dict")
                                    .allow_partial(true);
                                model.apply_from(&mut store).expect("Failed to load");
                            }
                            ModelSize::Medium => {
                                let mut model = MediumModel::<TestBackend>::new(&device);
                                let mut store = PytorchStore::from_file(checkpoint_path.clone())
                                    .with_top_level_key("model_state_dict")
                                    .allow_partial(true);
                                model.apply_from(&mut store).expect("Failed to load");
                            }
                            ModelSize::Large => {
                                let mut model = LargeModel::<TestBackend>::new(&device);
                                let mut store = PytorchStore::from_file(checkpoint_path.clone())
                                    .with_top_level_key("model_state_dict")
                                    .allow_partial(true);
                                model.apply_from(&mut store).expect("Failed to load");
                            }
                        }
                    });
            }
        }
    };
}

// Generate benchmarks for NdArray backend (always available)
bench_backend!(NdArrayBackend, ndarray_backend, "NdArray Backend (CPU)");

// Generate benchmarks for WGPU backend (if feature enabled)
#[cfg(feature = "wgpu")]
bench_backend!(WgpuBackend, wgpu_backend, "WGPU Backend (GPU)");

// Generate benchmarks for CUDA backend (if feature enabled)
#[cfg(feature = "cuda")]
bench_backend!(CudaBackend, cuda_backend, "CUDA Backend (NVIDIA GPU)");

// Generate benchmarks for Candle backend (if feature enabled)
#[cfg(feature = "candle")]
bench_backend!(CandleBackend, candle_backend, "Candle Backend");

// Generate benchmarks for LibTorch backend (if feature enabled)
#[cfg(feature = "tch")]
bench_backend!(TchBackend, tch_backend, "LibTorch Backend");

// Generate benchmarks for Metal backend (if feature enabled on macOS)
#[cfg(feature = "metal")]
bench_backend!(MetalBackend, metal_backend, "Metal Backend (Apple GPU)");
