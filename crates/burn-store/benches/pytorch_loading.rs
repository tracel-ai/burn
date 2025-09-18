#![recursion_limit = "256"]

//! Benchmarks for PyTorch model loading performance.
//!
//! This benchmark compares loading PyTorch models (.pth files) using:
//! 1. The old PyTorchFileRecorder from burn-import
//! 2. The new PytorchStore from burn-store
//!
//! Before running these benchmarks, generate the PyTorch model files:
//! ```bash
//! cd crates/burn-store
//! uv run benches/generate_pytorch_models.py
//! ```
//!
//! Then run the benchmarks:
//! ```bash
//! cargo bench --bench pytorch_loading
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
type MetalBackend = burn_wgpu::Metal;

// Simple model for basic benchmarks
#[derive(Module, Debug)]
struct SimpleModel<B: Backend> {
    linear1: nn::Linear<B>,
    linear2: nn::Linear<B>,
}

impl<B: Backend> SimpleModel<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            linear1: nn::LinearConfig::new(256, 512).init(device),
            linear2: nn::LinearConfig::new(512, 1024).init(device),
        }
    }
}

// Medium model with various layer types
#[derive(Module, Debug)]
struct MediumModel<B: Backend> {
    linear1: nn::Linear<B>,
    linear2: nn::Linear<B>,
    linear3: nn::Linear<B>,
    conv1: nn::conv::Conv2d<B>,
    conv2: nn::conv::Conv2d<B>,
}

impl<B: Backend> MediumModel<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            linear1: nn::LinearConfig::new(512, 1024).init(device),
            linear2: nn::LinearConfig::new(1024, 2048).init(device),
            linear3: nn::LinearConfig::new(2048, 4096).init(device),
            conv1: nn::conv::Conv2dConfig::new([3, 64], [3, 3])
                .with_padding(nn::PaddingConfig2d::Same)
                .init(device),
            conv2: nn::conv::Conv2dConfig::new([64, 128], [5, 5])
                .with_padding(nn::PaddingConfig2d::Same)
                .init(device),
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

            #[divan::bench_group(sample_count = 30)]
            mod small_model {
                use super::*;

                fn get_simple_model_paths() -> (PathBuf, PathBuf, PathBuf) {
                    let model_dir = check_model_files().expect("Model files not found");
                    (
                        model_dir.join("simple_model_state_dict.pth"),
                        model_dir.join("simple_model_checkpoint.pth"),
                        model_dir.join("simple_model_wrapped.pth"),
                    )
                }

                #[divan::bench(name = "old_recorder_state_dict")]
                fn old_recorder_state_dict(bencher: Bencher) {
                    let (state_dict_path, _, _) = get_simple_model_paths();
                    let file_size = fs::metadata(&state_dict_path).unwrap().len();

                    bencher
                        .counter(divan::counter::BytesCount::new(file_size))
                        .bench(|| {
                            let device: TestDevice = Default::default();
                            let recorder = PyTorchFileRecorder::<FullPrecisionSettings>::default();
                            let load_args = LoadArgs::new(state_dict_path.clone());
                            let record = recorder.load(load_args, &device).expect("Failed to load");
                            let _model =
                                SimpleModel::<TestBackend>::new(&device).load_record(record);
                        });
                }

                #[divan::bench(name = "new_store_state_dict")]
                fn new_store_state_dict(bencher: Bencher) {
                    let (state_dict_path, _, _) = get_simple_model_paths();
                    let file_size = fs::metadata(&state_dict_path).unwrap().len();

                    bencher
                        .counter(divan::counter::BytesCount::new(file_size))
                        .bench(|| {
                            let device: TestDevice = Default::default();
                            let mut model = SimpleModel::<TestBackend>::new(&device);
                            let mut store = PytorchStore::from_file(state_dict_path.clone())
                                .allow_partial(true);
                            model.apply_from(&mut store).expect("Failed to load");
                        });
                }

                #[divan::bench(name = "old_recorder_checkpoint")]
                fn old_recorder_checkpoint(bencher: Bencher) {
                    let (_, checkpoint_path, _) = get_simple_model_paths();
                    let file_size = fs::metadata(&checkpoint_path).unwrap().len();

                    bencher
                        .counter(divan::counter::BytesCount::new(file_size))
                        .bench(|| {
                            let device: TestDevice = Default::default();
                            let recorder = PyTorchFileRecorder::<FullPrecisionSettings>::default();
                            let load_args = LoadArgs::new(checkpoint_path.clone())
                                .with_top_level_key("model_state_dict");
                            let record = recorder.load(load_args, &device).expect("Failed to load");
                            let _model =
                                SimpleModel::<TestBackend>::new(&device).load_record(record);
                        });
                }

                #[divan::bench(name = "new_store_checkpoint")]
                fn new_store_checkpoint(bencher: Bencher) {
                    let (_, checkpoint_path, _) = get_simple_model_paths();
                    let file_size = fs::metadata(&checkpoint_path).unwrap().len();

                    bencher
                        .counter(divan::counter::BytesCount::new(file_size))
                        .bench(|| {
                            let device: TestDevice = Default::default();
                            let mut model = SimpleModel::<TestBackend>::new(&device);
                            let mut store = PytorchStore::from_file(checkpoint_path.clone())
                                .with_top_level_key("model_state_dict")
                                .allow_partial(true);
                            model.apply_from(&mut store).expect("Failed to load");
                        });
                }
            }

            #[divan::bench_group(sample_count = 20)]
            mod medium_model {
                use super::*;

                fn get_medium_model_paths() -> (PathBuf, PathBuf) {
                    let model_dir = check_model_files().expect("Model files not found");
                    (
                        model_dir.join("medium_model_state_dict.pth"),
                        model_dir.join("medium_model_checkpoint.pth"),
                    )
                }

                #[divan::bench(name = "old_recorder")]
                fn old_recorder_medium(bencher: Bencher) {
                    let (state_dict_path, _) = get_medium_model_paths();
                    let file_size = fs::metadata(&state_dict_path).unwrap().len();

                    bencher
                        .counter(divan::counter::BytesCount::new(file_size))
                        .bench(|| {
                            let device: TestDevice = Default::default();
                            let recorder = PyTorchFileRecorder::<FullPrecisionSettings>::default();
                            let load_args = LoadArgs::new(state_dict_path.clone());
                            let record = recorder.load(load_args, &device).expect("Failed to load");
                            let _model =
                                MediumModel::<TestBackend>::new(&device).load_record(record);
                        });
                }

                #[divan::bench(name = "new_store")]
                fn new_store_medium(bencher: Bencher) {
                    let (state_dict_path, _) = get_medium_model_paths();
                    let file_size = fs::metadata(&state_dict_path).unwrap().len();

                    bencher
                        .counter(divan::counter::BytesCount::new(file_size))
                        .bench(|| {
                            let device: TestDevice = Default::default();
                            let mut model = MediumModel::<TestBackend>::new(&device);
                            let mut store = PytorchStore::from_file(state_dict_path.clone())
                                .allow_partial(true);
                            model.apply_from(&mut store).expect("Failed to load");
                        });
                }
            }

            #[divan::bench_group(sample_count = 10)]
            mod large_model {
                use super::*;

                fn get_large_model_paths() -> (PathBuf, PathBuf) {
                    let model_dir = check_model_files().expect("Model files not found");
                    (
                        model_dir.join("large_model_state_dict.pth"),
                        model_dir.join("large_model_checkpoint.pth"),
                    )
                }

                #[divan::bench(name = "old_recorder")]
                fn old_recorder_large(bencher: Bencher) {
                    let (state_dict_path, _) = get_large_model_paths();
                    let file_size = fs::metadata(&state_dict_path).unwrap().len();

                    bencher
                        .counter(divan::counter::BytesCount::new(file_size))
                        .bench(|| {
                            let device: TestDevice = Default::default();
                            let recorder = PyTorchFileRecorder::<FullPrecisionSettings>::default();
                            let load_args = LoadArgs::new(state_dict_path.clone());
                            let record = recorder.load(load_args, &device).expect("Failed to load");
                            let _model =
                                LargeModel::<TestBackend>::new(&device).load_record(record);
                        });
                }

                #[divan::bench(name = "new_store")]
                fn new_store_large(bencher: Bencher) {
                    let (state_dict_path, _) = get_large_model_paths();
                    let file_size = fs::metadata(&state_dict_path).unwrap().len();

                    bencher
                        .counter(divan::counter::BytesCount::new(file_size))
                        .bench(|| {
                            let device: TestDevice = Default::default();
                            let mut model = LargeModel::<TestBackend>::new(&device);
                            let mut store = PytorchStore::from_file(state_dict_path.clone())
                                .allow_partial(true);
                            model.apply_from(&mut store).expect("Failed to load");
                        });
                }
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
