#![recursion_limit = "256"]

//! Benchmark comparing zero-copy vs copy loading modes for BurnpackStore.
//!
//! This benchmark measures the performance difference between:
//! - `zero_copy(false)` - Default mode, copies tensor data into new allocations
//! - `zero_copy(true)` - Zero-copy mode, slices tensor data without copying
//!
//! ## Understanding the Results
//!
//! **IMPORTANT**: For NdArray backend, you'll see similar allocation numbers because:
//! - NdArray uses `ndarray::ArrayD` which MUST own data as `Vec<T>`
//! - Even with zero-copy, the backend eventually copies data into its own format
//!
//! The zero-copy benefit is:
//! - **Without zero-copy**: File → Copy to heap (Bytes) → Copy to Vec (backend)
//! - **With zero-copy**: File → Zero-copy slice → Copy to Vec (backend)
//!
//! So zero-copy saves ONE memory copy at the store level. The `store_only_*` benchmarks
//! show the raw store performance without backend allocation overhead.
//!
//! GPU backends that can consume `Bytes` directly will show larger benefits.
//!
//! ## Running the benchmark
//!
//! Before running this benchmark, generate the model files:
//! ```bash
//! cd crates/burn-store
//! uv run benches/generate_unified_models.py
//! ```
//!
//! Then run the benchmark:
//! ```bash
//! cargo bench --bench zero_copy_loading
//! ```

use burn_core as burn;

use burn_core::module::Module;
use burn_core::prelude::*;
use burn_nn as nn;
use burn_store::{
    BurnpackStore, ModuleSnapshot, ModuleStore, PyTorchToBurnAdapter, SafetensorsStore,
};
use burn_tensor::{AllocationProperty, Bytes};
use divan::{AllocProfiler, Bencher};
use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

// Static storage for embedded model bytes (simulating include_bytes!)
static STATIC_MODEL_BYTES: OnceLock<&'static [u8]> = OnceLock::new();

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
        // Create a model with 20 layers - same as unified_loading benchmark
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

/// Get path to Burnpack model file
fn get_burnpack_path() -> PathBuf {
    get_model_dir().join("large_model.bpk")
}

/// Generate Burnpack file from existing SafeTensors file if needed
fn ensure_burnpack_file() {
    let bp_path = get_burnpack_path();
    let st_path = get_model_dir().join("large_model.safetensors");

    if bp_path.exists() {
        return;
    }

    if !st_path.exists() {
        panic!(
            "\n❌ SafeTensors model file not found!\n\
            \n\
            Please generate the model files first by running:\n\
            \n\
            cd crates/burn-store\n\
            uv run benches/generate_unified_models.py\n\
            \n\
            Expected file: {}\n",
            st_path.display()
        );
    }

    println!("⏳ Generating Burnpack file from SafeTensors...");

    type TestBackend = NdArrayBackend;
    let device = Default::default();

    // Load from SafeTensors
    let mut model = LargeModel::<TestBackend>::new(&device);
    let mut store = SafetensorsStore::from_file(&st_path).with_from_adapter(PyTorchToBurnAdapter);
    model
        .load_from(&mut store)
        .expect("Failed to load from SafeTensors");

    // Save as Burnpack
    let mut burnpack_store = BurnpackStore::from_file(&bp_path);
    model
        .save_into(&mut burnpack_store)
        .expect("Failed to save as Burnpack");

    println!("✅ Created Burnpack file: {}", bp_path.display());
}

/// Initialize static model bytes (simulating include_bytes! at runtime for benchmarks)
fn get_static_model_bytes() -> &'static [u8] {
    STATIC_MODEL_BYTES.get_or_init(|| {
        let bp_path = get_burnpack_path();
        let bytes = fs::read(&bp_path).expect("Failed to read Burnpack file");
        // Leak the bytes to get a 'static lifetime (acceptable for benchmarks)
        Box::leak(bytes.into_boxed_slice())
    })
}

fn main() {
    // Ensure Burnpack file exists
    ensure_burnpack_file();

    let bp_path = get_burnpack_path();
    let file_size = fs::metadata(&bp_path).unwrap().len() as f64 / 1_048_576.0;

    println!("✅ Found Burnpack model file:");
    println!("  Path: {}", bp_path.display());
    println!("  Size: {:.1} MB", file_size);
    println!();
    println!("🚀 Running zero-copy loading benchmarks...");
    println!();
    println!("Comparing loading modes:");
    println!("  1. file_copy        - from_file().zero_copy(false) - copies tensor data");
    println!("  2. file_zero_copy   - from_file().zero_copy(true)  - zero-copy via mmap");
    println!("  3. static_copy      - from_bytes() with Vec copy   - copies from static");
    println!("  4. static_zero_copy - from_static()                - zero-copy from static");
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

    // Pre-initialize static bytes before benchmarks
    let _ = get_static_model_bytes();

    divan::main();
}

// Macro to generate benchmarks for each backend
macro_rules! bench_backend {
    ($backend:ty, $mod_name:ident, $backend_name:literal) => {
        #[divan::bench_group(name = $backend_name, sample_count = 10)]
        mod $mod_name {
            use super::*;

            type TestBackend = $backend;
            type TestDevice = <TestBackend as Backend>::Device;

            /// File-based loading with copy mode (default)
            #[divan::bench]
            fn file_copy(bencher: Bencher) {
                let bp_path = get_burnpack_path();
                let file_size = fs::metadata(&bp_path).unwrap().len();

                bencher
                    .counter(divan::counter::BytesCount::new(file_size))
                    .bench(|| {
                        let device: TestDevice = Default::default();
                        let mut model = LargeModel::<TestBackend>::new(&device);
                        let mut store = BurnpackStore::from_file(&bp_path).zero_copy(false);
                        model.load_from(&mut store).expect("Failed to load");
                    });
            }

            /// File-based loading with zero-copy mode (mmap + bytes::Bytes)
            #[divan::bench]
            fn file_zero_copy(bencher: Bencher) {
                let bp_path = get_burnpack_path();
                let file_size = fs::metadata(&bp_path).unwrap().len();

                bencher
                    .counter(divan::counter::BytesCount::new(file_size))
                    .bench(|| {
                        let device: TestDevice = Default::default();
                        let mut model = LargeModel::<TestBackend>::new(&device);
                        let mut store = BurnpackStore::from_file(&bp_path).zero_copy(true);
                        model.load_from(&mut store).expect("Failed to load");
                    });
            }

            /// Static bytes with copy mode (simulating old behavior)
            #[divan::bench]
            fn static_copy(bencher: Bencher) {
                let static_bytes = get_static_model_bytes();
                let file_size = static_bytes.len() as u64;

                bencher
                    .counter(divan::counter::BytesCount::new(file_size))
                    .bench(|| {
                        let device: TestDevice = Default::default();
                        let mut model = LargeModel::<TestBackend>::new(&device);

                        // Simulate old behavior: copy static bytes to Vec, then load
                        let bytes = Bytes::from_bytes_vec(static_bytes.to_vec());
                        let mut store = BurnpackStore::from_bytes(Some(bytes)).zero_copy(false);
                        model.load_from(&mut store).expect("Failed to load");
                    });
            }

            /// Static bytes with zero-copy mode (new from_static)
            #[divan::bench]
            fn static_zero_copy(bencher: Bencher) {
                let static_bytes = get_static_model_bytes();
                let file_size = static_bytes.len() as u64;

                bencher
                    .counter(divan::counter::BytesCount::new(file_size))
                    .bench(|| {
                        let device: TestDevice = Default::default();
                        let mut model = LargeModel::<TestBackend>::new(&device);

                        // Zero-copy: use from_static which keeps data in .rodata
                        let mut store = BurnpackStore::from_static(static_bytes);
                        model.load_from(&mut store).expect("Failed to load");
                    });
            }

            /// In-memory shared bytes with zero-copy
            #[divan::bench]
            fn memory_shared_zero_copy(bencher: Bencher) {
                let static_bytes = get_static_model_bytes();
                let file_size = static_bytes.len() as u64;

                // Pre-create shared bytes outside the benchmark loop
                let shared = bytes::Bytes::from_static(static_bytes);

                bencher
                    .counter(divan::counter::BytesCount::new(file_size))
                    .bench(|| {
                        let device: TestDevice = Default::default();
                        let mut model = LargeModel::<TestBackend>::new(&device);

                        // Create Bytes from shared (cheap clone of Arc)
                        let bytes = Bytes::from_shared(shared.clone(), AllocationProperty::Other);
                        let mut store = BurnpackStore::from_bytes(Some(bytes)).zero_copy(true);
                        model.load_from(&mut store).expect("Failed to load");
                    });
            }
        }
    };
}

// =============================================================================
// Store-only benchmarks (no backend allocation overhead)
// These show the TRUE zero-copy benefit at the store level
// =============================================================================

#[divan::bench_group(name = "Store Only (no backend)", sample_count = 10)]
mod store_only {
    use super::*;

    /// File-based store with copy mode - measures store overhead only
    #[divan::bench]
    fn file_copy(bencher: Bencher) {
        let bp_path = get_burnpack_path();
        let file_size = fs::metadata(&bp_path).unwrap().len();

        bencher
            .counter(divan::counter::BytesCount::new(file_size))
            .bench(|| {
                let mut store = BurnpackStore::from_file(&bp_path).zero_copy(false);
                // Just iterate through all tensor snapshots, calling to_data() on each
                // This forces the store to read and materialize all tensor data
                let snapshots = store.get_all_snapshots().expect("Failed to get snapshots");
                for (_, snapshot) in snapshots {
                    let _data = snapshot.to_data().expect("Failed to get tensor data");
                }
            });
    }

    /// File-based store with zero-copy mode - measures store overhead only
    #[divan::bench]
    fn file_zero_copy(bencher: Bencher) {
        let bp_path = get_burnpack_path();
        let file_size = fs::metadata(&bp_path).unwrap().len();

        bencher
            .counter(divan::counter::BytesCount::new(file_size))
            .bench(|| {
                let mut store = BurnpackStore::from_file(&bp_path).zero_copy(true);
                let snapshots = store.get_all_snapshots().expect("Failed to get snapshots");
                for (_, snapshot) in snapshots {
                    let _data = snapshot.to_data().expect("Failed to get tensor data");
                }
            });
    }

    /// Static bytes with copy mode - measures store overhead only
    #[divan::bench]
    fn static_copy(bencher: Bencher) {
        let static_bytes = get_static_model_bytes();
        let file_size = static_bytes.len() as u64;

        bencher
            .counter(divan::counter::BytesCount::new(file_size))
            .bench(|| {
                // Simulate old behavior: copy static bytes to Vec
                let bytes = Bytes::from_bytes_vec(static_bytes.to_vec());
                let mut store = BurnpackStore::from_bytes(Some(bytes)).zero_copy(false);
                let snapshots = store.get_all_snapshots().expect("Failed to get snapshots");
                for (_, snapshot) in snapshots {
                    let _data = snapshot.to_data().expect("Failed to get tensor data");
                }
            });
    }

    /// Static bytes with zero-copy mode - measures store overhead only
    #[divan::bench]
    fn static_zero_copy(bencher: Bencher) {
        let static_bytes = get_static_model_bytes();
        let file_size = static_bytes.len() as u64;

        bencher
            .counter(divan::counter::BytesCount::new(file_size))
            .bench(|| {
                let mut store = BurnpackStore::from_static(static_bytes);
                let snapshots = store.get_all_snapshots().expect("Failed to get snapshots");
                for (_, snapshot) in snapshots {
                    let _data = snapshot.to_data().expect("Failed to get tensor data");
                }
            });
    }
}

// =============================================================================
// Full model loading benchmarks (includes backend allocation)
// =============================================================================

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
