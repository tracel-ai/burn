#![recursion_limit = "256"]

//! Unified benchmark comparing all loading methods:
//! - SafetensorsStore (new)
//! - SafetensorsFileRecorder (old)
//! - PytorchStore (new)
//! - PyTorchFileRecorder (old)
//!
//! Before running this benchmark, generate the model files:
//! ```bash
//! cd crates/burn-store
//! uv run benches/generate_unified_models.py
//! ```
//!
//! Then run the benchmark:
//! ```bash
//! cargo bench --bench unified_loading
//! ```

use burn_core::module::Module;
use burn_core::nn;
use burn_core::prelude::*;
use burn_core::record::{FullPrecisionSettings, Recorder};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};
use burn_import::safetensors::SafetensorsFileRecorder;
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, PytorchStore, SafetensorsStore};
use divan::{AllocProfiler, Bencher};
use std::fs;
use std::path::PathBuf;

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

// Backend type aliases
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

// Deep model matching the Python script structure
#[derive(Module, Debug)]
struct DeepModel<B: Backend> {
    layers: Vec<nn::Linear<B>>,
}

impl<B: Backend> DeepModel<B> {
    fn new(device: &B::Device) -> Self {
        let mut layers = Vec::new();
        let mut layer_configs = Vec::new();

        // First 10 layers: gradually increase from 512 to 1024
        for i in 0..10 {
            let in_size = if i == 0 { 512 } else { 512 + i * 50 };
            let out_size = 512 + (i + 1) * 50;
            layer_configs.push((in_size, out_size));
        }

        // Middle 35 layers: even larger layers for more parameters
        for i in 0..35 {
            if i % 3 == 0 {
                layer_configs.push((2048, 3072));
            } else if i % 3 == 1 {
                layer_configs.push((3072, 2048));
            } else {
                layer_configs.push((2048, 2048));
            }
        }

        // Last 10 layers: gradually decrease back to 512
        let mut prev_out_size = 2048;
        for i in 0..10 {
            let in_size = prev_out_size;
            let mut out_size = 1024 - (i + 1) * 50;
            if out_size < 512 {
                out_size = 512;
            }
            layer_configs.push((in_size, out_size));
            prev_out_size = out_size;
        }

        // Create the layers
        for (in_size, out_size) in layer_configs {
            layers.push(nn::LinearConfig::new(in_size, out_size).init(device));
        }

        Self { layers }
    }
}

/// Get the path to the model files
fn get_model_dir() -> PathBuf {
    std::env::temp_dir().join("unified_bench_models")
}

/// Get paths to the model files
fn get_model_paths() -> (PathBuf, PathBuf) {
    let dir = get_model_dir();
    (
        dir.join("deep_model.safetensors"),
        dir.join("deep_model.pt"),
    )
}

/// Check if model files exist
fn check_model_files() -> Result<(), String> {
    let (st_path, pt_path) = get_model_paths();

    if !st_path.exists() || !pt_path.exists() {
        return Err(format!(
            "\n❌ Model files not found!\n\
            \n\
            Please generate the model files first by running:\n\
            \n\
            cd crates/burn-store\n\
            uv run benches/generate_unified_models.py\n\
            \n\
            Expected files:\n\
            - {}\n\
            - {}\n",
            st_path.display(),
            pt_path.display()
        ));
    }

    Ok(())
}

fn main() {
    // Check if model files exist before running benchmarks
    match check_model_files() {
        Ok(()) => {
            let (st_path, pt_path) = get_model_paths();
            let st_size = fs::metadata(&st_path).unwrap().len() as f64 / 1_048_576.0;
            let pt_size = fs::metadata(&pt_path).unwrap().len() as f64 / 1_048_576.0;

            println!("✅ Found model files:");
            println!("  SafeTensors: {} ({:.1} MB)", st_path.display(), st_size);
            println!("  PyTorch: {} ({:.1} MB)", pt_path.display(), pt_size);
            println!();
            println!("🚀 Running unified loading benchmarks...");
            println!();
            println!("Comparing 4 loading methods:");
            println!("  1. SafetensorsStore (new)");
            println!("  2. SafetensorsFileRecorder (old)");
            println!("  3. PytorchStore (new)");
            println!("  4. PyTorchFileRecorder (old)");
            println!();
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
        #[divan::bench_group(name = $backend_name, sample_count = 10)]
        mod $mod_name {
            use super::*;

            type TestBackend = $backend;
            type TestDevice = <TestBackend as Backend>::Device;

            #[divan::bench]
            fn safetensors_store(bencher: Bencher) {
                let (st_path, _) = get_model_paths();
                let file_size = fs::metadata(&st_path).unwrap().len();

                bencher
                    .counter(divan::counter::BytesCount::new(file_size))
                    .bench(|| {
                        let device: TestDevice = Default::default();
                        let mut model = DeepModel::<TestBackend>::new(&device);
                        let mut store = SafetensorsStore::from_file(st_path.clone())
                            .with_from_adapter(PyTorchToBurnAdapter);
                        model.apply_from(&mut store).expect("Failed to load");
                    });
            }

            #[divan::bench]
            fn safetensors_recorder(bencher: Bencher) {
                let (st_path, _) = get_model_paths();
                let file_size = fs::metadata(&st_path).unwrap().len();

                bencher
                    .counter(divan::counter::BytesCount::new(file_size))
                    .bench(|| {
                        let device: TestDevice = Default::default();
                        let recorder = SafetensorsFileRecorder::<FullPrecisionSettings>::default();
                        let record = recorder
                            .load(st_path.clone().into(), &device)
                            .expect("Failed to load");
                        let _model = DeepModel::<TestBackend>::new(&device).load_record(record);
                    });
            }

            #[divan::bench]
            fn pytorch_store(bencher: Bencher) {
                let (_, pt_path) = get_model_paths();
                let file_size = fs::metadata(&pt_path).unwrap().len();

                bencher
                    .counter(divan::counter::BytesCount::new(file_size))
                    .bench(|| {
                        let device: TestDevice = Default::default();
                        let mut model = DeepModel::<TestBackend>::new(&device);
                        let mut store = PytorchStore::from_file(pt_path.clone())
                            .with_top_level_key("model_state_dict")
                            .allow_partial(true);
                        model.apply_from(&mut store).expect("Failed to load");
                    });
            }

            #[divan::bench]
            fn pytorch_recorder(bencher: Bencher) {
                let (_, pt_path) = get_model_paths();
                let file_size = fs::metadata(&pt_path).unwrap().len();

                bencher
                    .counter(divan::counter::BytesCount::new(file_size))
                    .bench(|| {
                        let device: TestDevice = Default::default();
                        let recorder = PyTorchFileRecorder::<FullPrecisionSettings>::default();
                        let load_args =
                            LoadArgs::new(pt_path.clone()).with_top_level_key("model_state_dict");
                        let record = recorder.load(load_args, &device).expect("Failed to load");
                        let _model = DeepModel::<TestBackend>::new(&device).load_record(record);
                    });
            }
        }
    };
}

// Generate benchmarks for each backend
bench_backend!(NdArrayBackend, ndarray_backend, "NdArray Backend (CPU)");

#[cfg(feature = "wgpu")]
bench_backend!(WgpuBackend, wgpu_backend, "WGPU Backend (GPU)");

#[cfg(feature = "cuda")]
bench_backend!(CudaBackend, cuda_backend, "CUDA Backend (NVIDIA GPU)");

#[cfg(feature = "candle")]
bench_backend!(CandleBackend, candle_backend, "Candle Backend");

#[cfg(feature = "tch")]
bench_backend!(TchBackend, tch_backend, "LibTorch Backend");

#[cfg(feature = "metal")]
bench_backend!(MetalBackend, metal_backend, "Metal Backend (Apple GPU)");
