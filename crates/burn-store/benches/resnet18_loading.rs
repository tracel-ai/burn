//! Benchmark for ResNet18 loading to verify lazy loading memory usage.
//!
//! resnet18.pth is pytorch's legacy file format.
//!
//! This benchmark loads a ResNet18 model and materializes all tensors
//! to ensure memory usage stays reasonable with lazy loading.
//!
//! Run the benchmark:
//! ```bash
//! cargo bench --bench resnet18_loading
//! ```

use burn_store::pytorch::PytorchReader;
use divan::{AllocProfiler, Bencher};
use std::path::PathBuf;

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    // Check if ResNet18 file exists
    let path = resnet18_path();
    if !path.exists() {
        eprintln!("❌ ResNet18 model not found!");
        eprintln!("");
        eprintln!("Please download it first by running:");
        eprintln!("  python benches/download_resnet18.py");
        eprintln!("");
        eprintln!("Or if you don't have Python/PyTorch installed:");
        eprintln!("  uv run benches/download_resnet18.py");
        eprintln!("");
        eprintln!("Expected location: {}", path.display());
        std::process::exit(1);
    }

    // Verify file size is reasonable
    let metadata = std::fs::metadata(&path).expect("Failed to read file metadata");
    let size_mb = metadata.len() as f64 / 1_048_576.0;

    if size_mb < 40.0 || size_mb > 50.0 {
        eprintln!(
            "⚠️ Warning: ResNet18 file size ({:.1} MB) seems unusual",
            size_mb
        );
        eprintln!("Expected size is around 45 MB");
    }

    println!("✅ Found ResNet18 model at: {}", path.display());
    println!("📦 File size: {:.1} MB", size_mb);
    println!("📊 Running ResNet18 loading benchmarks...\n");

    // Run divan benchmarks
    divan::main();
}

/// Get the path to ResNet18 model file
fn resnet18_path() -> PathBuf {
    // First try to read from the path file created by download script
    let temp_dir = std::env::temp_dir();
    let config_file = temp_dir.join("burn_resnet18_benchmark").join("path.txt");

    if config_file.exists() {
        if let Ok(path_str) = std::fs::read_to_string(&config_file) {
            let path = PathBuf::from(path_str.trim());
            if path.exists() {
                return path;
            }
        }
    }

    // Fallback to default location
    temp_dir
        .join("burn_resnet18_benchmark")
        .join("resnet18.pth")
}

#[divan::bench(sample_count = 10)]
fn load_resnet18_metadata(bencher: Bencher) {
    let path = resnet18_path();

    bencher.bench_local(|| {
        let reader = PytorchReader::new(&path).expect("Failed to load ResNet18");
        let metadata = reader.metadata();

        // Just access metadata without materializing tensors
        assert_eq!(metadata.tensor_count, 122);
    });
}

#[divan::bench(sample_count = 5)]
fn load_resnet18_materialize_all(bencher: Bencher) {
    let path = resnet18_path();

    bencher.bench_local(|| {
        let reader = PytorchReader::new(&path).expect("Failed to load ResNet18");
        let keys = reader.keys();

        let mut total_bytes = 0usize;

        // Materialize all tensors one by one
        for key in &keys {
            let tensor = reader.get(key).expect("Failed to get tensor");
            // Materialize the tensor data
            let _data = tensor.to_data().expect("Failed to materialize tensor data");
            total_bytes += tensor.data_len();
        }

        // Verify we processed all the data
        assert!(total_bytes > 40_000_000); // Should be ~45MB
    });
}

#[divan::bench(sample_count = 5)]
fn load_resnet18_materialize_sequential(bencher: Bencher) {
    let path = resnet18_path();

    bencher.bench_local(|| {
        let reader = PytorchReader::new(&path).expect("Failed to load ResNet18");
        let keys = reader.keys();

        // Materialize tensors one at a time, letting previous ones be dropped
        // This simulates processing tensors sequentially without keeping all in memory
        for key in &keys {
            let tensor = reader.get(key).expect("Failed to get tensor");
            let data = tensor.to_data().expect("Failed to materialize tensor data");

            // Do minimal work with the data to prevent optimization
            let sum = match data.dtype {
                burn_tensor::DType::F32 => data
                    .as_slice::<f32>()
                    .map(|s| s.iter().sum::<f32>())
                    .unwrap_or(0.0) as f64,
                burn_tensor::DType::F64 => data
                    .as_slice::<f64>()
                    .map(|s| s.iter().sum::<f64>())
                    .unwrap_or(0.0),
                _ => 0.0,
            };

            // Use the sum to prevent dead code elimination
            std::hint::black_box(sum);
        }
    });
}

#[divan::bench(sample_count = 10)]
fn load_resnet18_largest_tensor(bencher: Bencher) {
    let path = resnet18_path();

    bencher.bench_local(|| {
        let reader = PytorchReader::new(&path).expect("Failed to load ResNet18");

        // Find and materialize only the largest tensor
        // This tests peak memory for a single tensor operation
        let keys = reader.keys();
        let mut largest_key = String::new();
        let mut largest_size = 0usize;

        for key in &keys {
            let tensor = reader.get(key).expect("Failed to get tensor");
            let size = tensor.data_len();
            if size > largest_size {
                largest_size = size;
                largest_key = key.clone();
            }
        }

        // Materialize the largest tensor
        let tensor = reader
            .get(&largest_key)
            .expect("Failed to get largest tensor");
        let _data = tensor.to_data().expect("Failed to materialize tensor data");

        assert!(largest_size > 9_000_000); // Should be ~9MB for layer4.0.conv2.weight
    });
}

#[divan::bench(sample_count = 10)]
fn load_resnet18_memory_profile(bencher: Bencher) {
    let path = resnet18_path();

    bencher
        .with_inputs(|| path.clone())
        .bench_local_values(|path| {
            let reader = PytorchReader::new(&path).expect("Failed to load ResNet18");
            let keys = reader.keys();

            let mut peak_single_tensor = 0usize;
            let mut total_data = 0usize;

            // Process each tensor and track memory
            for key in &keys {
                let tensor = reader.get(key).expect("Failed to get tensor");
                let tensor_size = tensor.data_len();

                // Track largest single tensor
                if tensor_size > peak_single_tensor {
                    peak_single_tensor = tensor_size;
                }

                // Materialize the tensor
                let data = tensor.to_data().expect("Failed to materialize tensor data");
                total_data += tensor_size;

                // Drop data immediately to test lazy loading memory efficiency
                drop(data);
            }

            // Return stats for verification
            (peak_single_tensor, total_data)
        });
}
