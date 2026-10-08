// The LibTorch bench group exists to compare against the deprecated backend.
#![cfg_attr(feature = "tch", allow(deprecated))]

//! Unified benchmark comparing all loading methods:
//! - BurnpackStore (lazy burnpack loading)
//! - ModuleRecord (the record API, reading the same burnpack format)
//! - SafetensorsStore
//! - PytorchStore
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

use burn_core as burn;

use burn_core::module::Module;
use burn_core::prelude::*;
use burn_core::store::ModuleRecord;
use burn_nn as nn;
use burn_store::{
    BurnpackStore, ModuleSnapshot, PyTorchToBurnAdapter, PytorchStore, SafetensorsStore,
};
use divan::{AllocProfiler, Bencher};
use std::fs;
use std::path::{Path, PathBuf};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

// Use the same LargeModel as other benchmarks for fair comparison
#[derive(Module, Debug)]
struct LargeModel {
    layers: Vec<nn::Linear>,
}

impl LargeModel {
    fn new(device: &Device) -> Self {
        let mut layers = Vec::new();
        // Create a model with 20 layers - same as safetensor_loading benchmark
        for i in 0..20 {
            let in_size = if i == 0 { 1024 } else { 2048 };
            layers.push(nn::LinearConfig::new(in_size, 2048).init(device));
        }
        Self { layers }
    }
}

/// Get the path to the model files
fn get_model_dir() -> PathBuf {
    std::env::temp_dir().join("simple_bench_models")
}

/// Generate the Burnpack and ModuleRecord files from the existing SafeTensors file
fn generate_burn_formats(st_path: &Path, bp_path: &Path, record_path: &Path) {
    let device = Device::flex();

    // Load the model from SafeTensors
    let mut model = LargeModel::new(&device);
    let mut store = SafetensorsStore::from_file(st_path).with_from_adapter(PyTorchToBurnAdapter);
    model
        .load_from(&mut store)
        .expect("Failed to load from SafeTensors");

    // Save as Burnpack
    if !bp_path.exists() {
        println!("  Creating Burnpack file...");
        let mut burnpack_store = BurnpackStore::from_file(bp_path);
        model
            .save_into(&mut burnpack_store)
            .expect("Failed to save as Burnpack");
    }

    // Save through the record API
    if !record_path.exists() {
        println!("  Creating ModuleRecord file...");
        model
            .save_file(record_path)
            .expect("Failed to save with ModuleRecord");
    }
}

/// Get paths to the model files
fn get_model_paths() -> (PathBuf, PathBuf, PathBuf, PathBuf) {
    let dir = get_model_dir();
    (
        dir.join("large_model.bpk"),
        dir.join("large_model_record.bpk"),
        dir.join("large_model.safetensors"),
        dir.join("large_model.pt"),
    )
}

/// Check if model files exist
fn check_model_files() -> Result<(), String> {
    let (_, _, st_path, pt_path) = get_model_paths();

    // Only the safetensors and pytorch files are required; the burnpack files are generated
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
            let (bp_path, record_path, st_path, pt_path) = get_model_paths();

            // First, generate the burnpack files if they don't exist
            if !bp_path.exists() || !record_path.exists() {
                println!("⏳ Generating Burnpack and ModuleRecord files from SafeTensors...");
                generate_burn_formats(&st_path, &bp_path, &record_path);
            }

            let bp_size = fs::metadata(&bp_path)
                .ok()
                .map(|m| m.len() as f64 / 1_048_576.0);
            let record_size = fs::metadata(&record_path)
                .ok()
                .map(|m| m.len() as f64 / 1_048_576.0);
            let st_size = fs::metadata(&st_path).unwrap().len() as f64 / 1_048_576.0;
            let pt_size = fs::metadata(&pt_path).unwrap().len() as f64 / 1_048_576.0;

            println!("✅ Found model files:");
            if let Some(size) = bp_size {
                println!("  Burnpack: {} ({:.1} MB)", bp_path.display(), size);
            }
            if let Some(size) = record_size {
                println!("  ModuleRecord: {} ({:.1} MB)", record_path.display(), size);
            }
            println!("  SafeTensors: {} ({:.1} MB)", st_path.display(), st_size);
            println!("  PyTorch: {} ({:.1} MB)", pt_path.display(), pt_size);
            println!();
            println!("🚀 Running unified loading benchmarks...");
            println!();
            println!("Comparing 4 loading methods:");
            println!("  1. BurnpackStore (lazy burnpack loading)");
            println!("  2. ModuleRecord (record API, same burnpack format)");
            println!("  3. SafetensorsStore");
            println!("  4. PytorchStore");
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
            eprintln!("{}", msg);
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
                let (bp_path, _, _, _) = get_model_paths();
                let file_size = fs::metadata(&bp_path).unwrap().len();

                bencher
                    .counter(divan::counter::BytesCount::new(file_size))
                    .bench(|| {
                        let device = $device;
                        let mut model = LargeModel::new(&device);
                        let mut store = BurnpackStore::from_file(bp_path.clone());
                        model.load_from(&mut store).expect("Failed to load");
                    });
            }

            #[divan::bench]
            fn module_record(bencher: Bencher) {
                let (_, record_path, _, _) = get_model_paths();
                let file_size = fs::metadata(&record_path).unwrap().len();

                bencher
                    .counter(divan::counter::BytesCount::new(file_size))
                    .bench(|| {
                        let device = $device;
                        let model = LargeModel::new(&device);
                        model
                            .load_record(ModuleRecord::load(&record_path).expect("Failed to load"));
                    });
            }

            #[divan::bench]
            fn safetensors_store(bencher: Bencher) {
                let (_, _, st_path, _) = get_model_paths();
                let file_size = fs::metadata(&st_path).unwrap().len();

                bencher
                    .counter(divan::counter::BytesCount::new(file_size))
                    .bench(|| {
                        let device = $device;
                        let mut model = LargeModel::new(&device);
                        let mut store = SafetensorsStore::from_file(st_path.clone())
                            .with_from_adapter(PyTorchToBurnAdapter);
                        model.load_from(&mut store).expect("Failed to load");
                    });
            }

            #[divan::bench]
            fn pytorch_store(bencher: Bencher) {
                let (_, _, _, pt_path) = get_model_paths();
                let file_size = fs::metadata(&pt_path).unwrap().len();

                bencher
                    .counter(divan::counter::BytesCount::new(file_size))
                    .bench(|| {
                        let device: Device = $device.into();
                        let mut model = LargeModel::new(&device);
                        let mut store = PytorchStore::from_file(pt_path.clone())
                            .with_top_level_key("model_state_dict")
                            .allow_partial(true);
                        model.load_from(&mut store).expect("Failed to load");
                    });
            }
        }
    };
}

// Generate benchmarks for each backend
bench_backend!(Device::flex(), flex_backend, "Flex Backend (CPU)");

#[cfg(feature = "wgpu")]
bench_backend!(
    Device::wgpu(Default::default()),
    wgpu_backend,
    "WGPU Backend (GPU)"
);

#[cfg(feature = "cuda")]
bench_backend!(
    CudaDevice::default(),
    cuda_backend,
    "CUDA Backend (NVIDIA GPU)"
);

#[cfg(feature = "tch")]
bench_backend!(Device::libtorch(), tch_backend, "LibTorch Backend");

#[cfg(feature = "metal")]
bench_backend!(
    Device::wgpu(Default::default()),
    metal_backend,
    "Metal Backend (Apple GPU)"
);
