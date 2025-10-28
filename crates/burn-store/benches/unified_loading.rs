#![recursion_limit = "256"]

//! Unified benchmark comparing all loading methods:
//! - BurnpackStore (new native format)
//! - NamedMpkFileRecorder (old native format)
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

use burn_core as burn;

use burn_core::module::Module;
use burn_core::prelude::*;
use burn_core::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
// use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};
// use burn_import::safetensors::SafetensorsFileRecorder;
use burn_nn as nn;
use burn_store::{
    BurnpackStore, ModuleSnapshot, PyTorchToBurnAdapter, PytorchStore, SafetensorsStore,
};
use divan::{AllocProfiler, Bencher};
use std::fs;
use std::path::{Path, PathBuf};

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

/// Generate Burnpack and NamedMpk files from existing SafeTensors file
fn generate_burn_formats(st_path: &Path, bp_path: &Path, mpk_path: &Path) {
    type TestBackend = NdArrayBackend;
    let device = Default::default();

    // Load the model from SafeTensors
    let mut model = LargeModel::<TestBackend>::new(&device);
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

    // Save as NamedMpk
    if !mpk_path.exists() {
        println!("  Creating NamedMpk file...");
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
        model
            .save_file(mpk_path, &recorder)
            .expect("Failed to save as NamedMpk");
    }
}

/// Get paths to the model files
fn get_model_paths() -> (PathBuf, PathBuf, PathBuf, PathBuf) {
    let dir = get_model_dir();
    (
        dir.join("large_model.bpk"),
        dir.join("large_model.mpk"),
        dir.join("large_model.safetensors"),
        dir.join("large_model.pt"),
    )
}

/// Check if model files exist
fn check_model_files() -> Result<(), String> {
    let (_, _, st_path, pt_path) = get_model_paths();

    // For now, only check safetensors and pytorch files (will generate burnpack/mpk later)
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
            let (bp_path, mpk_path, st_path, pt_path) = get_model_paths();

            // First, generate Burnpack and MPK files if they don't exist
            if !bp_path.exists() || !mpk_path.exists() {
                println!("⏳ Generating Burnpack and NamedMpk files from SafeTensors...");
                generate_burn_formats(&st_path, &bp_path, &mpk_path);
            }

            let bp_size = fs::metadata(&bp_path)
                .ok()
                .map(|m| m.len() as f64 / 1_048_576.0);
            let mpk_size = fs::metadata(&mpk_path)
                .ok()
                .map(|m| m.len() as f64 / 1_048_576.0);
            let st_size = fs::metadata(&st_path).unwrap().len() as f64 / 1_048_576.0;
            let pt_size = fs::metadata(&pt_path).unwrap().len() as f64 / 1_048_576.0;

            println!("✅ Found model files:");
            if let Some(size) = bp_size {
                println!("  Burnpack: {} ({:.1} MB)", bp_path.display(), size);
            }
            if let Some(size) = mpk_size {
                println!("  NamedMpk: {} ({:.1} MB)", mpk_path.display(), size);
            }
            println!("  SafeTensors: {} ({:.1} MB)", st_path.display(), st_size);
            println!("  PyTorch: {} ({:.1} MB)", pt_path.display(), pt_size);
            println!();
            println!("🚀 Running unified loading benchmarks...");
            println!();
            println!("Comparing 6 loading methods:");
            println!("  1. BurnpackStore (new native format - lazy loading)");
            println!("  2. NamedMpkFileRecorder (old native format - loads all to memory)");
            println!("  3. SafetensorsStore (new)");
            println!("  4. SafetensorsFileRecorder (old)");
            println!("  5. PytorchStore (new)");
            println!("  6. PyTorchFileRecorder (old)");
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
            fn burnpack_store(bencher: Bencher) {
                let (bp_path, _, _, _) = get_model_paths();
                let file_size = fs::metadata(&bp_path).unwrap().len();

                bencher
                    .counter(divan::counter::BytesCount::new(file_size))
                    .bench(|| {
                        let device: TestDevice = Default::default();
                        let mut model = LargeModel::<TestBackend>::new(&device);
                        let mut store = BurnpackStore::from_file(bp_path.clone());
                        model.load_from(&mut store).expect("Failed to load");
                    });
            }

            #[divan::bench]
            fn namedmpk_recorder(bencher: Bencher) {
                let (_, mpk_path, _, _) = get_model_paths();
                let file_size = fs::metadata(&mpk_path).unwrap().len();

                bencher
                    .counter(divan::counter::BytesCount::new(file_size))
                    .bench(|| {
                        let device: TestDevice = Default::default();
                        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
                        let record = recorder
                            .load(mpk_path.clone().into(), &device)
                            .expect("Failed to load");
                        let _model = LargeModel::<TestBackend>::new(&device).load_record(record);
                    });
            }

            #[divan::bench]
            fn safetensors_store(bencher: Bencher) {
                let (_, _, st_path, _) = get_model_paths();
                let file_size = fs::metadata(&st_path).unwrap().len();

                bencher
                    .counter(divan::counter::BytesCount::new(file_size))
                    .bench(|| {
                        let device: TestDevice = Default::default();
                        let mut model = LargeModel::<TestBackend>::new(&device);
                        let mut store = SafetensorsStore::from_file(st_path.clone())
                            .with_from_adapter(PyTorchToBurnAdapter);
                        model.load_from(&mut store).expect("Failed to load");
                    });
            }

            // #[divan::bench]
            // fn safetensors_recorder(bencher: Bencher) {
            //     let (_, _, st_path, _) = get_model_paths();
            //     let file_size = fs::metadata(&st_path).unwrap().len();

            //     bencher
            //         .counter(divan::counter::BytesCount::new(file_size))
            //         .bench(|| {
            //             let device: TestDevice = Default::default();
            //             let recorder = SafetensorsFileRecorder::<FullPrecisionSettings>::default();
            //             let record = recorder
            //                 .load(st_path.clone().into(), &device)
            //                 .expect("Failed to load");
            //             let _model = LargeModel::<TestBackend>::new(&device).load_record(record);
            //         });
            // }

            #[divan::bench]
            fn pytorch_store(bencher: Bencher) {
                let (_, _, _, pt_path) = get_model_paths();
                let file_size = fs::metadata(&pt_path).unwrap().len();

                bencher
                    .counter(divan::counter::BytesCount::new(file_size))
                    .bench(|| {
                        let device: TestDevice = Default::default();
                        let mut model = LargeModel::<TestBackend>::new(&device);
                        let mut store = PytorchStore::from_file(pt_path.clone())
                            .with_top_level_key("model_state_dict")
                            .allow_partial(true);
                        model.load_from(&mut store).expect("Failed to load");
                    });
            }

            // #[divan::bench]
            // fn pytorch_recorder(bencher: Bencher) {
            //     let (_, _, _, pt_path) = get_model_paths();
            //     let file_size = fs::metadata(&pt_path).unwrap().len();

            //     bencher
            //         .counter(divan::counter::BytesCount::new(file_size))
            //         .bench(|| {
            //             let device: TestDevice = Default::default();
            //             let recorder = PyTorchFileRecorder::<FullPrecisionSettings>::default();
            //             let load_args =
            //                 LoadArgs::new(pt_path.clone()).with_top_level_key("model_state_dict");
            //             let record = recorder.load(load_args, &device).expect("Failed to load");
            //             let _model = LargeModel::<TestBackend>::new(&device).load_record(record);
            //         });
            // }
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
