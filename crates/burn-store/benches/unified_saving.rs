#![recursion_limit = "256"]

//! Unified benchmark comparing all saving methods:
//! - BurnpackStore (new native format)
//! - NamedMpkFileRecorder (old native format)
//! - SafetensorsStore (new)
//!
//! Before running this benchmark, ensure the directory exists:
//! ```bash
//! mkdir -p /tmp/simple_bench_models
//! ```
//!
//! Then run the benchmark:
//! ```bash
//! cargo bench --bench unified_saving
//! ```

use burn_core::module::Module;
use burn_core::prelude::*;
use burn_core::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn_nn as nn;
use burn_store::{BurnpackStore, ModuleSnapshot, SafetensorsStore};
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
type MetalBackend = burn_wgpu::Metal;

// Use the same LargeModel as other benchmarks for fair comparison
#[derive(Module, Debug)]
struct LargeModel<B: Backend> {
    layers: Vec<nn::Linear<B>>,
}

impl<B: Backend> LargeModel<B> {
    fn new(device: &B::Device) -> Self {
        let mut layers = Vec::new();
        // Create a model with 20 layers - same as loading benchmarks
        for i in 0..20 {
            let in_size = if i == 0 { 1024 } else { 2048 };
            layers.push(nn::LinearConfig::new(in_size, 2048).init(device));
        }
        Self { layers }
    }
}

/// Get the path to the output directory
fn get_output_dir() -> PathBuf {
    std::env::temp_dir().join("simple_bench_models_saving")
}

/// Ensure output directory exists
fn ensure_output_dir() -> Result<(), String> {
    let dir = get_output_dir();
    if !dir.exists() {
        fs::create_dir_all(&dir)
            .map_err(|e| format!("Failed to create output directory: {}", e))?;
    }
    Ok(())
}

fn main() {
    match ensure_output_dir() {
        Ok(()) => {
            println!("✅ Output directory ready: {}", get_output_dir().display());
            println!();
            println!("🚀 Running unified saving benchmarks...");
            println!();
            println!("Comparing 3 saving methods:");
            println!("  1. BurnpackStore (new native format)");
            println!("  2. NamedMpkFileRecorder (old native format)");
            println!("  3. SafetensorsStore (new)");
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
            eprintln!("❌ {}", msg);
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
            fn burnpack_store(bencher: Bencher) {
                bencher.bench(|| {
                    let device: TestDevice = Default::default();
                    let model = LargeModel::<TestBackend>::new(&device);
                    let output_path = get_output_dir().join("test_burnpack.burnpack");
                    let mut store = BurnpackStore::from_file(output_path.clone());
                    model
                        .collect_to(&mut store)
                        .expect("Failed to save with BurnpackStore");
                    // Clean up
                    let _ = fs::remove_file(output_path);
                });
            }

            #[divan::bench]
            fn namedmpk_recorder(bencher: Bencher) {
                bencher.bench(|| {
                    let device: TestDevice = Default::default();
                    let model = LargeModel::<TestBackend>::new(&device);
                    let output_path = get_output_dir().join("test_namedmpk.mpk");
                    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
                    model
                        .save_file(output_path.clone(), &recorder)
                        .expect("Failed to save with NamedMpkFileRecorder");
                    // Clean up
                    let _ = fs::remove_file(output_path);
                });
            }

            #[divan::bench]
            fn safetensors_store(bencher: Bencher) {
                bencher.bench(|| {
                    let device: TestDevice = Default::default();
                    let model = LargeModel::<TestBackend>::new(&device);
                    let output_path = get_output_dir().join("test_safetensors_store.safetensors");
                    let mut store = SafetensorsStore::from_file(output_path.clone());
                    model
                        .collect_to(&mut store)
                        .expect("Failed to save with SafetensorsStore");
                    // Clean up
                    let _ = fs::remove_file(output_path);
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
