extern crate alloc;

use burn::prelude::*;
use serde::Deserialize;
use std::fs;
use std::path::Path;

#[cfg(feature = "wgpu")]
pub type MyBackend = burn::backend::Wgpu;

#[cfg(feature = "ndarray")]
pub type MyBackend = burn::backend::NdArray<f32>;

#[cfg(feature = "tch")]
pub type MyBackend = burn::backend::LibTorch<f32>;

#[cfg(feature = "metal")]
pub type MyBackend = burn::backend::Metal;

// Include the generated model
include!(concat!(env!("OUT_DIR"), "/model/silero_vad.rs"));

/// Test case from reference outputs
#[derive(Debug, Deserialize)]
struct TestCase {
    test_name: String,
    #[allow(dead_code)]
    chunk_index: i32,
    #[allow(dead_code)]
    start_sample: i32,
    input_samples: Vec<f32>,
    expected_output: f32,
    #[allow(dead_code)]
    state_after: Vec<Vec<Vec<f32>>>,
}

/// Reference outputs structure
#[derive(Debug, Deserialize)]
struct ReferenceOutputs {
    sample_rate: i64,
    #[allow(dead_code)]
    chunk_size: usize,
    #[allow(dead_code)]
    audio_length_samples: usize,
    test_cases: Vec<TestCase>,
}

/// Run a single test case and return (passed, actual_output, expected_output)
fn run_test_case(
    model: &Model<MyBackend>,
    device: &<MyBackend as Backend>::Device,
    test_case: &TestCase,
    sample_rate: i64,
) -> (bool, f32, f32) {
    // Create input tensor from test case samples
    let input_data: Vec<f32> = test_case.input_samples.clone();
    let input = Tensor::<MyBackend, 1>::from_floats(input_data.as_slice(), device)
        .reshape([1, test_case.input_samples.len()]);

    // Initialize state to zeros
    let state = Tensor::<MyBackend, 3>::zeros([2, 1, 128], device);

    // Run inference
    let (output, _state_out) = model.forward(input, sample_rate, state);

    // Get the output probability
    let actual_output: f32 = output.into_scalar();
    let expected_output = test_case.expected_output;

    // Compare with tolerance (neural networks have small floating point differences)
    let tolerance = 0.01; // 1% tolerance
    let passed = (actual_output - expected_output).abs() < tolerance;

    (passed, actual_output, expected_output)
}

fn main() {
    println!("========================================");
    println!("Silero VAD Model Test Suite");
    println!("========================================\n");

    // Check if artifacts exist
    let artifacts_dir = Path::new("artifacts");
    if !artifacts_dir.exists() {
        eprintln!("Error: artifacts directory not found!");
        eprintln!("Please run get_model.py first to download the model.");
        eprintln!("Example: uv run get_model.py");
        std::process::exit(1);
    }

    // Check if reference outputs exist
    let reference_path = artifacts_dir.join("reference_outputs.json");
    if !reference_path.exists() {
        eprintln!("Error: reference_outputs.json not found!");
        eprintln!("Please run: uv run get_model.py");
        std::process::exit(1);
    }

    // Load reference outputs
    println!("Loading reference outputs...");
    let reference_json = fs::read_to_string(&reference_path).expect("Failed to read reference outputs");
    let reference: ReferenceOutputs =
        serde_json::from_str(&reference_json).expect("Failed to parse reference outputs");
    println!(
        "  Loaded {} test cases (sample rate: {} Hz)\n",
        reference.test_cases.len(),
        reference.sample_rate
    );

    // Initialize the model
    println!("Initializing Silero VAD model...");
    let device = Default::default();
    let model: Model<MyBackend> = Model::default();
    println!("  Model initialized\n");

    // Run tests
    println!("Running test cases...");
    println!("{:-<60}", "");

    let mut passed_count = 0;
    let mut failed_count = 0;

    for test_case in &reference.test_cases {
        let (passed, actual, expected) = run_test_case(&model, &device, test_case, reference.sample_rate);

        if passed {
            println!(
                "  [PASS] {}: output={:.6} (expected={:.6})",
                test_case.test_name, actual, expected
            );
            passed_count += 1;
        } else {
            println!(
                "  [FAIL] {}: output={:.6} (expected={:.6}, diff={:.6})",
                test_case.test_name,
                actual,
                expected,
                (actual - expected).abs()
            );
            failed_count += 1;
        }
    }

    println!("{:-<60}", "");
    println!();

    // Summary
    println!("========================================");
    println!("Test Summary");
    println!("========================================");
    println!("  Total tests: {}", passed_count + failed_count);
    println!("  Passed: {}", passed_count);
    println!("  Failed: {}", failed_count);
    println!();

    if failed_count == 0 {
        println!("All tests passed!");
        println!("The Burn model produces outputs matching ONNX Runtime.");
    } else {
        println!("Some tests failed!");
        println!("The Burn model outputs differ from ONNX Runtime.");
        std::process::exit(1);
    }
}
