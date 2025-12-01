//! Benchmark comparing mmap vs non-mmap ONNX loading.
//!
//! This benchmark uses real models from Hugging Face to demonstrate
//! the memory benefits of memory-mapped loading.
//!
//! ## Setup
//!
//! Before running this benchmark, download the models:
//! ```bash
//! cd crates/onnx-ir
//! uv run benches/generate_bench_model.py
//! ```
//!
//! ## Running
//!
//! ```bash
//! # With mmap (default):
//! cargo bench --bench mmap_loading
//!
//! # Without mmap:
//! cargo bench --bench mmap_loading --no-default-features
//! ```
//!
//! ## What this measures
//!
//! - `parse_file`: Load from file path (uses mmap when available)
//! - `parse_bytes`: Load from in-memory bytes (no mmap, requires pre-reading)
//! - `parse_reader`: Load from a reader (no mmap, reads into memory)
//!
//! The key insight is that `parse_file` with mmap should show:
//! - Lower memory allocation (tensor data stays in mmap'd region)
//! - Potentially faster startup (no upfront copy of entire file)

use divan::{AllocProfiler, Bencher};
use std::fs;
use std::path::PathBuf;

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

/// Get the path to the MiniLM model (~86 MB)
fn get_minilm_path() -> PathBuf {
    std::env::temp_dir()
        .join("onnx_ir_bench_models")
        .join("all-minilm-l6-v2_opset16.onnx")
}

/// Get the path to the CLIP Vision model (~336 MB)
fn get_clip_path() -> PathBuf {
    std::env::temp_dir()
        .join("onnx_ir_bench_models")
        .join("clip-vit-b-32-vision_opset16.onnx")
}

/// Get the path to a smaller fixture model for quick tests
fn get_small_model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("large_constants.onnx")
}

/// Check if benchmark models exist
fn check_models() -> Result<(), String> {
    let minilm_path = get_minilm_path();
    let clip_path = get_clip_path();
    let small_path = get_small_model_path();

    if !small_path.exists() {
        return Err(format!(
            "Small test fixture not found: {}\n\
            This is unexpected - check your repository integrity.",
            small_path.display()
        ));
    }

    if !minilm_path.exists() || !clip_path.exists() {
        return Err(format!(
            "\n❌ Benchmark models not found!\n\
            \n\
            Please download the models first by running:\n\
            \n\
            cd crates/onnx-ir\n\
            uv run benches/generate_bench_model.py\n\
            \n\
            This downloads:\n\
            - MiniLM (~86 MB) - sentence transformer\n\
            - CLIP Vision (~336 MB) - vision encoder\n\
            \n\
            Expected files:\n\
            - {}\n\
            - {}\n",
            minilm_path.display(),
            clip_path.display()
        ));
    }

    Ok(())
}

fn main() {
    match check_models() {
        Ok(()) => {
            let minilm_path = get_minilm_path();
            let clip_path = get_clip_path();
            let small_path = get_small_model_path();

            let minilm_size = fs::metadata(&minilm_path).unwrap().len() as f64 / 1_048_576.0;
            let clip_size = fs::metadata(&clip_path).unwrap().len() as f64 / 1_048_576.0;
            let small_size = fs::metadata(&small_path).unwrap().len() as f64 / 1024.0;

            println!("✅ Found model files:");
            println!(
                "  MiniLM: {} ({:.1} MB)",
                minilm_path.display(),
                minilm_size
            );
            println!(
                "  CLIP Vision: {} ({:.1} MB)",
                clip_path.display(),
                clip_size
            );
            println!(
                "  Small fixture: {} ({:.1} KB)",
                small_path.display(),
                small_size
            );
            println!();

            #[cfg(feature = "mmap")]
            println!("🗺️  mmap feature: ENABLED");
            #[cfg(not(feature = "mmap"))]
            println!("📦 mmap feature: DISABLED");

            println!();
            println!("🚀 Running ONNX loading benchmarks...");
            println!();
            println!("Comparing loading methods:");
            println!("  1. parse_file  - Load from file path (uses mmap when enabled)");
            println!("  2. parse_bytes - Load from pre-read bytes (no mmap)");
            println!("  3. parse_reader - Load from reader (no mmap)");
            println!();

            divan::main();
        }
        Err(msg) => {
            eprintln!("{}", msg);
            std::process::exit(1);
        }
    }
}

/// Benchmarks using CLIP Vision (~336 MB)
#[divan::bench_group(name = "CLIP Vision (336 MB)", sample_count = 5)]
mod clip_vision {
    use super::*;
    use onnx_ir::OnnxGraphBuilder;

    #[divan::bench]
    fn parse_file(bencher: Bencher) {
        let path = get_clip_path();
        let file_size = fs::metadata(&path).unwrap().len();

        bencher
            .counter(divan::counter::BytesCount::new(file_size))
            .bench(|| {
                OnnxGraphBuilder::new()
                    .parse_file(&path)
                    .expect("Failed to parse ONNX model")
            });
    }

    #[divan::bench]
    fn parse_bytes(bencher: Bencher) {
        let path = get_clip_path();
        let bytes = fs::read(&path).expect("Failed to read file");
        let file_size = bytes.len() as u64;

        bencher
            .counter(divan::counter::BytesCount::new(file_size))
            .bench(|| {
                OnnxGraphBuilder::new()
                    .parse_bytes(&bytes)
                    .expect("Failed to parse ONNX model")
            });
    }

    #[divan::bench]
    fn parse_reader(bencher: Bencher) {
        let path = get_clip_path();
        let file_size = fs::metadata(&path).unwrap().len();

        bencher
            .counter(divan::counter::BytesCount::new(file_size))
            .bench(|| {
                let file = fs::File::open(&path).expect("Failed to open file");
                OnnxGraphBuilder::new()
                    .parse_reader(file)
                    .expect("Failed to parse ONNX model")
            });
    }
}

/// Benchmarks using MiniLM (~86 MB)
#[divan::bench_group(name = "MiniLM (86 MB)", sample_count = 10)]
mod minilm {
    use super::*;
    use onnx_ir::OnnxGraphBuilder;

    #[divan::bench]
    fn parse_file(bencher: Bencher) {
        let path = get_minilm_path();
        let file_size = fs::metadata(&path).unwrap().len();

        bencher
            .counter(divan::counter::BytesCount::new(file_size))
            .bench(|| {
                OnnxGraphBuilder::new()
                    .parse_file(&path)
                    .expect("Failed to parse ONNX model")
            });
    }

    #[divan::bench]
    fn parse_bytes(bencher: Bencher) {
        let path = get_minilm_path();
        let bytes = fs::read(&path).expect("Failed to read file");
        let file_size = bytes.len() as u64;

        bencher
            .counter(divan::counter::BytesCount::new(file_size))
            .bench(|| {
                OnnxGraphBuilder::new()
                    .parse_bytes(&bytes)
                    .expect("Failed to parse ONNX model")
            });
    }

    #[divan::bench]
    fn parse_reader(bencher: Bencher) {
        let path = get_minilm_path();
        let file_size = fs::metadata(&path).unwrap().len();

        bencher
            .counter(divan::counter::BytesCount::new(file_size))
            .bench(|| {
                let file = fs::File::open(&path).expect("Failed to open file");
                OnnxGraphBuilder::new()
                    .parse_reader(file)
                    .expect("Failed to parse ONNX model")
            });
    }
}

/// Benchmarks using the small (1MB) fixture model
#[divan::bench_group(name = "Small Model (1MB)", sample_count = 50)]
mod small_model {
    use super::*;
    use onnx_ir::OnnxGraphBuilder;

    #[divan::bench]
    fn parse_file(bencher: Bencher) {
        let path = get_small_model_path();
        let file_size = fs::metadata(&path).unwrap().len();

        bencher
            .counter(divan::counter::BytesCount::new(file_size))
            .bench(|| {
                OnnxGraphBuilder::new()
                    .parse_file(&path)
                    .expect("Failed to parse ONNX model")
            });
    }

    #[divan::bench]
    fn parse_bytes(bencher: Bencher) {
        let path = get_small_model_path();
        let bytes = fs::read(&path).expect("Failed to read file");
        let file_size = bytes.len() as u64;

        bencher
            .counter(divan::counter::BytesCount::new(file_size))
            .bench(|| {
                OnnxGraphBuilder::new()
                    .parse_bytes(&bytes)
                    .expect("Failed to parse ONNX model")
            });
    }

    #[divan::bench]
    fn parse_reader(bencher: Bencher) {
        let path = get_small_model_path();
        let file_size = fs::metadata(&path).unwrap().len();

        bencher
            .counter(divan::counter::BytesCount::new(file_size))
            .bench(|| {
                let file = fs::File::open(&path).expect("Failed to open file");
                OnnxGraphBuilder::new()
                    .parse_reader(file)
                    .expect("Failed to parse ONNX model")
            });
    }
}
