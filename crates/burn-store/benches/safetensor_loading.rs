#![recursion_limit = "256"]

use burn::nn;
use burn_core::module::Module;
use burn_core::prelude::*;
use burn_core::record::{FullPrecisionSettings, Recorder};
use burn_import::safetensors::SafetensorsFileRecorder;
use burn_store::ModuleSnapshot;
use burn_store::safetensors::SafetensorsStore;
use divan::{AllocProfiler, Bencher};
use std::path::PathBuf;
use std::{fs, path::Path};
use tempfile::tempdir;

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

fn create_test_file<B: Backend, M: Module<B>>(path: &Path, model: M) {
    let mut store = SafetensorsStore::from_file(path);
    model.collect_to(&mut store).expect("Failed to save model");
}

fn main() {
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

                fn setup_simple_model_file() -> (tempfile::TempDir, PathBuf, u64) {
                    let temp_dir = tempdir().unwrap();
                    let file_path = temp_dir.path().join("simple_model.safetensors");
                    let device: TestDevice = Default::default();
                    let model = SimpleModel::<TestBackend>::new(&device);
                    create_test_file(&file_path, model);
                    let file_size = fs::metadata(&file_path).unwrap().len();
                    (temp_dir, file_path, file_size)
                }

                #[divan::bench(name = "old_recorder")]
                fn old_recorder_simple(bencher: Bencher) {
                    let (_temp_dir, file_path, file_size) = setup_simple_model_file();

                    bencher
                        .counter(divan::counter::BytesCount::new(file_size))
                        .bench(|| {
                            let device: TestDevice = Default::default();
                            let recorder =
                                SafetensorsFileRecorder::<FullPrecisionSettings>::default();
                            let record = recorder
                                .load(file_path.clone().into(), &device)
                                .expect("Failed to load");
                            let _model =
                                SimpleModel::<TestBackend>::new(&device).load_record(record);
                        });
                }

                #[divan::bench(name = "new_store")]
                fn new_store_simple(bencher: Bencher) {
                    let (_temp_dir, file_path, file_size) = setup_simple_model_file();

                    bencher
                        .counter(divan::counter::BytesCount::new(file_size))
                        .bench(|| {
                            let device: TestDevice = Default::default();
                            let mut model = SimpleModel::<TestBackend>::new(&device);
                            let mut store = SafetensorsStore::from_file(file_path.clone());
                            model.apply_from(&mut store).expect("Failed to load");
                        });
                }
            }

            #[divan::bench_group(sample_count = 20)]
            mod medium_model {
                use super::*;

                fn setup_medium_model_file() -> (tempfile::TempDir, PathBuf, u64) {
                    let temp_dir = tempdir().unwrap();
                    let file_path = temp_dir.path().join("medium_model.safetensors");
                    let device: TestDevice = Default::default();
                    let model = MediumModel::<TestBackend>::new(&device);
                    create_test_file(&file_path, model);
                    let file_size = fs::metadata(&file_path).unwrap().len();
                    (temp_dir, file_path, file_size)
                }

                #[divan::bench(name = "old_recorder")]
                fn old_recorder_medium(bencher: Bencher) {
                    let (_temp_dir, file_path, file_size) = setup_medium_model_file();

                    bencher
                        .counter(divan::counter::BytesCount::new(file_size))
                        .bench(|| {
                            let device: TestDevice = Default::default();
                            let recorder =
                                SafetensorsFileRecorder::<FullPrecisionSettings>::default();
                            let record = recorder
                                .load(file_path.clone().into(), &device)
                                .expect("Failed to load");
                            let _model =
                                MediumModel::<TestBackend>::new(&device).load_record(record);
                        });
                }

                #[divan::bench(name = "new_store")]
                fn new_store_medium(bencher: Bencher) {
                    let (_temp_dir, file_path, file_size) = setup_medium_model_file();

                    bencher
                        .counter(divan::counter::BytesCount::new(file_size))
                        .bench(|| {
                            let device: TestDevice = Default::default();
                            let mut model = MediumModel::<TestBackend>::new(&device);
                            let mut store = SafetensorsStore::from_file(file_path.clone());
                            model.apply_from(&mut store).expect("Failed to load");
                        });
                }
            }

            #[divan::bench_group(sample_count = 10)]
            mod large_model {
                use super::*;

                fn setup_large_model_file() -> (tempfile::TempDir, PathBuf, u64) {
                    let temp_dir = tempdir().unwrap();
                    let file_path = temp_dir.path().join("large_model.safetensors");
                    let device: TestDevice = Default::default();
                    let model = LargeModel::<TestBackend>::new(&device);
                    create_test_file(&file_path, model);
                    let file_size = fs::metadata(&file_path).unwrap().len();
                    (temp_dir, file_path, file_size)
                }

                #[divan::bench(name = "old_recorder")]
                fn old_recorder_large(bencher: Bencher) {
                    let (_temp_dir, file_path, file_size) = setup_large_model_file();

                    bencher
                        .counter(divan::counter::BytesCount::new(file_size))
                        .bench(|| {
                            let device: TestDevice = Default::default();
                            let recorder =
                                SafetensorsFileRecorder::<FullPrecisionSettings>::default();
                            let record = recorder
                                .load(file_path.clone().into(), &device)
                                .expect("Failed to load");
                            let _model =
                                LargeModel::<TestBackend>::new(&device).load_record(record);
                        });
                }

                #[divan::bench(name = "new_store")]
                fn new_store_large(bencher: Bencher) {
                    let (_temp_dir, file_path, file_size) = setup_large_model_file();

                    bencher
                        .counter(divan::counter::BytesCount::new(file_size))
                        .bench(|| {
                            let device: TestDevice = Default::default();
                            let mut model = LargeModel::<TestBackend>::new(&device);
                            let mut store = SafetensorsStore::from_file(file_path.clone());
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
