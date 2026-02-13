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
// Zero-copy verification (proves operations use static region data)
// =============================================================================

/// Verify that zero-copy loading actually uses data from the static region.
/// This runs once at startup to prove correctness before benchmarking.
#[divan::bench_group(name = "Zero-Copy Verification", sample_count = 1)]
mod verification {
    use super::*;
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    /// Verify zero-copy: tensor storage is borrowed (not owned)
    #[divan::bench]
    fn verify_storage_is_borrowed() {
        let static_bytes = get_static_model_bytes();

        // Load model with zero-copy from static bytes
        let device = Default::default();
        let mut model = LargeModel::<B>::new(&device);
        let mut store = BurnpackStore::from_static(static_bytes);
        model.load_from(&mut store).expect("Failed to load");

        // Get the first layer's weight tensor and verify it uses borrowed storage
        let weight = model.layers[0].weight.val();
        // .into_primitive() returns TensorPrimitive<B>, .tensor() extracts B::FloatTensorPrimitive
        let ndarray_tensor = weight.into_primitive().tensor();

        // Verify the storage is borrowed (zero-copy from static region)
        assert!(
            ndarray_tensor.is_borrowed(),
            "ZERO-COPY FAILURE: Tensor storage is NOT borrowed. \
             Data was copied instead of being zero-copy!"
        );

        println!("✅ Verified: Tensor storage is borrowed (zero-copy from static region)");
    }

    /// Verify ALL layers use borrowed (zero-copy) storage.
    /// This is the key proof that loaded weights point to static memory.
    #[divan::bench]
    fn verify_all_layers_borrowed() {
        let static_bytes = get_static_model_bytes();

        // Load model with zero-copy
        let device = Default::default();
        let mut model = LargeModel::<B>::new(&device);
        let mut store = BurnpackStore::from_static(static_bytes);
        model.load_from(&mut store).expect("Failed to load");

        // Check ALL layers have borrowed storage
        let mut total_elements = 0usize;
        for (i, layer) in model.layers.iter().enumerate() {
            let weight = layer.weight.val();
            total_elements += weight.shape().num_elements();

            assert!(
                weight.into_primitive().tensor().is_borrowed(),
                "Layer {} weight should be borrowed (zero-copy)",
                i
            );
        }

        let total_mb = (total_elements * 4) as f64 / 1_048_576.0;
        println!(
            "✅ Verified: All {} layers use borrowed storage",
            model.layers.len()
        );
        println!(
            "   - Model size: {:.2} MB - all pointing to static region",
            total_mb
        );
    }

    /// Verify data is readable and correct using sum().into_scalar().
    /// Note: sum() triggers COW copy, so this shows ops work correctly on zero-copy data.
    #[divan::bench]
    fn verify_ops_produce_correct_results() {
        let static_bytes = get_static_model_bytes();

        let device = Default::default();
        let mut model = LargeModel::<B>::new(&device);
        let mut store = BurnpackStore::from_static(static_bytes);
        model.load_from(&mut store).expect("Failed to load");

        // Compute sum of first layer weight - proves data is valid
        let weight = model.layers[0].weight.val();
        let sum: f32 = weight.sum().into_scalar();

        assert!(sum.is_finite(), "Sum should be finite");
        println!("✅ Verified: Operations on zero-copy data produce valid results");
        println!("   - First layer sum: {:.4}", sum);
    }

    /// Verify operations produce correct results on zero-copy data
    #[divan::bench]
    fn verify_operations_on_static_data() {
        let static_bytes = get_static_model_bytes();

        // Load model with zero-copy
        let device = Default::default();
        let mut model = LargeModel::<B>::new(&device);
        let mut store = BurnpackStore::from_static(static_bytes);
        model.load_from(&mut store).expect("Failed to load");

        // Perform operations on the loaded weights
        let weight = model.layers[0].weight.val();
        let shape = weight.shape();

        // Test 1: Sum should be finite (not NaN or Inf)
        let sum: f32 = weight.clone().sum().to_data().to_vec().unwrap()[0];
        assert!(
            sum.is_finite(),
            "Operation failed: sum is not finite ({})",
            sum
        );

        // Test 2: Matrix multiply with itself transposed (W @ W.T)
        let transposed = weight.clone().transpose();
        let matmul_result = weight.clone().matmul(transposed);
        let matmul_sum: f32 = matmul_result.sum().to_data().to_vec().unwrap()[0];
        assert!(
            matmul_sum.is_finite(),
            "Matmul failed: result sum is not finite ({})",
            matmul_sum
        );

        // Test 3: Element-wise operations
        let doubled = weight.clone() * 2.0;
        let doubled_sum: f32 = doubled.sum().to_data().to_vec().unwrap()[0];
        assert!(
            (doubled_sum - sum * 2.0).abs() < 1e-3,
            "Element-wise op failed: doubled_sum ({}) != sum*2 ({})",
            doubled_sum,
            sum * 2.0
        );

        println!("✅ Verified: Operations on zero-copy data produce correct results");
        println!("   - Weight shape: {:?}", shape.as_slice());
        println!("   - Sum: {:.4}", sum);
        println!("   - Matmul result sum: {:.4}", matmul_sum);
    }

    /// Compare zero-copy vs copy: verify both produce identical results
    #[divan::bench]
    fn verify_copy_vs_zero_copy_equality() {
        let static_bytes = get_static_model_bytes();
        let device: <B as Backend>::Device = Default::default();

        // Load with zero-copy
        let mut model_zc = LargeModel::<B>::new(&device);
        let mut store_zc = BurnpackStore::from_static(static_bytes);
        model_zc
            .load_from(&mut store_zc)
            .expect("Failed to load zero-copy");

        // Load with copy (simulate old behavior)
        let mut model_copy = LargeModel::<B>::new(&device);
        let bytes = Bytes::from_bytes_vec(static_bytes.to_vec());
        let mut store_copy = BurnpackStore::from_bytes(Some(bytes)).zero_copy(false);
        model_copy
            .load_from(&mut store_copy)
            .expect("Failed to load copy");

        // Compare weights from both models
        for (i, (layer_zc, layer_copy)) in model_zc
            .layers
            .iter()
            .zip(model_copy.layers.iter())
            .enumerate()
        {
            let weight_zc = layer_zc.weight.val();
            let weight_copy = layer_copy.weight.val();

            // Check shapes match
            assert_eq!(
                weight_zc.shape(),
                weight_copy.shape(),
                "Layer {} weight shapes don't match",
                i
            );

            // Check values match (using sum as a proxy)
            let sum_zc: f32 = weight_zc.clone().sum().to_data().to_vec().unwrap()[0];
            let sum_copy: f32 = weight_copy.clone().sum().to_data().to_vec().unwrap()[0];
            assert!(
                (sum_zc - sum_copy).abs() < 1e-6,
                "Layer {} weight sums don't match: zero-copy={}, copy={}",
                i,
                sum_zc,
                sum_copy
            );
        }

        println!(
            "✅ Verified: Zero-copy and copy loading produce identical results for all {} layers",
            model_zc.layers.len()
        );
    }
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
                for snapshot in snapshots.values() {
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
                for snapshot in snapshots.values() {
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
                for snapshot in snapshots.values() {
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
                for snapshot in snapshots.values() {
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

#[cfg(feature = "tch")]
bench_backend!(TchBackend, tch_backend, "LibTorch Backend");

#[cfg(feature = "metal")]
bench_backend!(MetalBackend, metal_backend, "Metal Backend (Apple GPU)");
