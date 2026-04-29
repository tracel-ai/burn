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
use burn_core as burn;

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
use burn_core::tensor::FlexDevice;

#[cfg(any(feature = "wgpu", feature = "metal"))]
use burn_core::tensor::WgpuDevice;

#[cfg(feature = "cuda")]
use burn_core::tensor::CudaDevice;

#[cfg(feature = "tch")]
use burn_core::tensor::LibTorchDevice;

// Use the same LargeModel as other benchmarks for fair comparison
#[derive(Module, Debug)]
struct LargeModel {
    layers: Vec<nn::Linear>,
}

impl LargeModel {
    fn new(device: &Device) -> Self {
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
            println!("  - Flex (CPU)");
            #[cfg(feature = "wgpu")]
            println!("  - WGPU (GPU)");
            #[cfg(feature = "cuda")]
            println!("  - CUDA (NVIDIA GPU)");
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
    ($device:expr, $mod_name:ident, $backend_name:literal) => {
        #[divan::bench_group(name = $backend_name, sample_count = 10)]
        mod $mod_name {
            use super::*;

            #[divan::bench]
            fn burnpack_store(bencher: Bencher) {
                bencher.bench(|| {
                    let device: Device = $device.into();
                    let model = LargeModel::new(&device);
                    let output_path = get_output_dir().join("test_burnpack.bpk");
                    let mut store = BurnpackStore::from_file(output_path.clone()).overwrite(true);
                    model
                        .save_into(&mut store)
                        .expect("Failed to save with BurnpackStore");
                    // Clean up
                    let _ = fs::remove_file(output_path);
                });
            }

            #[divan::bench]
            fn namedmpk_recorder(bencher: Bencher) {
                bencher.bench(|| {
                    let device: Device = $device.into();
                    let model = LargeModel::new(&device);
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
                    let device: Device = $device.into();
                    let model = LargeModel::new(&device);
                    let output_path = get_output_dir().join("test_safetensors_store.safetensors");
                    let mut store = SafetensorsStore::from_file(output_path.clone());
                    model
                        .save_into(&mut store)
                        .expect("Failed to save with SafetensorsStore");
                    // Clean up
                    let _ = fs::remove_file(output_path);
                });
            }
        }
    };
}

// Generate benchmarks for each backend
bench_backend!(FlexDevice, ndarray_backend, "NdArray Backend (CPU)");

#[cfg(feature = "wgpu")]
bench_backend!(WgpuDevice::default(), wgpu_backend, "WGPU Backend (GPU)");

#[cfg(feature = "cuda")]
bench_backend!(
    CudaDevice::default(),
    cuda_backend,
    "CUDA Backend (NVIDIA GPU)"
);

#[cfg(feature = "tch")]
bench_backend!(LibTorchDevice::default(), tch_backend, "LibTorch Backend");

#[cfg(feature = "metal")]
bench_backend!(
    WgpuDevice::default(),
    metal_backend,
    "Metal Backend (Apple GPU)"
);
