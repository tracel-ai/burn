extern crate alloc;

use burn::module::Param;
use burn::prelude::*;
use burn::record::*;

use burn_import::pytorch::PyTorchFileRecorder;
use std::path::Path;
use std::time::Instant;

#[cfg(feature = "wgpu")]
pub type MyBackend = burn::backend::Wgpu;

#[cfg(feature = "ndarray")]
pub type MyBackend = burn::backend::NdArray<f32>;

#[cfg(feature = "tch")]
pub type MyBackend = burn::backend::LibTorch<f32>;

#[cfg(feature = "metal")]
pub type MyBackend = burn::backend::Metal;

// Include the generated model
include!(concat!(env!("OUT_DIR"), "/model/rf_detr_small.rs"));

/// Test data structure matching the PyTorch saved format
/// RF-DETR has two outputs: dets (bounding boxes) and labels (class scores)
#[derive(Debug, Module)]
struct TestData<B: Backend> {
    input: Param<Tensor<B, 4>>,
    output_dets: Param<Tensor<B, 3>>,
    output_labels: Param<Tensor<B, 3>>,
}

fn main() {
    println!("========================================");
    println!("RF-DETR Small Model Test");
    println!("========================================\n");

    // Check if artifacts exist
    let artifacts_dir = Path::new("artifacts");
    if !artifacts_dir.exists() {
        eprintln!("Error: artifacts directory not found!");
        eprintln!("Please run get_model.py first to download the model.");
        eprintln!("Example: uv run --python 3.11 get_model.py");
        std::process::exit(1);
    }

    // Check if model file exists
    let model_file = artifacts_dir.join("rf_detr_small.onnx");
    let test_data_file = artifacts_dir.join("rf_detr_small_test_data.pt");

    if !model_file.exists() {
        eprintln!("Error: Model file not found!");
        eprintln!("Please run: uv run --python 3.11 get_model.py");
        std::process::exit(1);
    }

    if !test_data_file.exists() {
        eprintln!("Error: Test data file not found!");
        eprintln!("Please run: uv run --python 3.11 get_model.py");
        std::process::exit(1);
    }

    // Initialize the model
    println!("Initializing RF-DETR Small model...");
    let start = Instant::now();
    let device = Default::default();
    let model: Model<MyBackend> = Model::default();
    let init_time = start.elapsed();
    println!("  Model initialized in {:.2?}", init_time);

    // Save model structure to file
    let model_txt_path = artifacts_dir.join("rf_detr_small_model.txt");
    println!(
        "\nSaving model structure to {}...",
        model_txt_path.display()
    );
    let model_str = format!("{}", model);
    std::fs::write(&model_txt_path, &model_str).expect("Failed to write model structure to file");
    println!("  Model structure saved");

    // Load test data from PyTorch file
    println!("\nLoading test data from {}...", test_data_file.display());
    let start = Instant::now();
    let test_data: TestDataRecord<MyBackend> = PyTorchFileRecorder::<FullPrecisionSettings>::new()
        .load(test_data_file.into(), &device)
        .expect("Failed to load test data");
    let load_time = start.elapsed();
    println!("  Data loaded in {:.2?}", load_time);

    // Get the input tensor from test data
    let input = test_data.input.val();
    let input_shape = input.shape();
    println!("  Loaded input tensor with shape: {:?}", input_shape.dims);

    // Get the reference outputs from test data
    let reference_dets = test_data.output_dets.val();
    let reference_labels = test_data.output_labels.val();
    println!(
        "  Loaded reference dets with shape: {:?}",
        reference_dets.shape().dims
    );
    println!(
        "  Loaded reference labels with shape: {:?}",
        reference_labels.shape().dims
    );

    // Run inference with the loaded input
    println!("\nRunning model inference with test input...");
    let start = Instant::now();
    let (output_dets, output_labels) = model.forward(input);
    let inference_time = start.elapsed();
    println!("  Inference completed in {:.2?}", inference_time);

    // Display output shapes
    println!("\nModel outputs:");
    println!("  Dets shape: {:?}", output_dets.shape().dims);
    println!("  Labels shape: {:?}", output_labels.shape().dims);

    // Compare dets output
    println!("\nComparing dets output with reference data...");
    if output_dets
        .clone()
        .all_close(reference_dets.clone(), Some(1e-4), Some(1e-4))
    {
        println!("  Dets output matches reference data within tolerance (1e-4)!");
    } else {
        println!("  Dets output differs from reference data!");
        let diff = output_dets.clone() - reference_dets.clone();
        let abs_diff = diff.abs();
        let max_diff = abs_diff.clone().max().into_scalar();
        let mean_diff = abs_diff.mean().into_scalar();
        println!("  Maximum absolute difference: {:.6}", max_diff);
        println!("  Mean absolute difference: {:.6}", mean_diff);
    }

    // Compare labels output
    println!("\nComparing labels output with reference data...");
    if output_labels
        .clone()
        .all_close(reference_labels.clone(), Some(1e-4), Some(1e-4))
    {
        println!("  Labels output matches reference data within tolerance (1e-4)!");
    } else {
        println!("  Labels output differs from reference data!");
        let diff = output_labels.clone() - reference_labels.clone();
        let abs_diff = diff.abs();
        let max_diff = abs_diff.clone().max().into_scalar();
        let mean_diff = abs_diff.mean().into_scalar();
        println!("  Maximum absolute difference: {:.6}", max_diff);
        println!("  Mean absolute difference: {:.6}", mean_diff);
    }

    println!("\n========================================");
    println!("Model test completed!");
    println!("========================================");
    println!();
    println!("RF-DETR is a transformer-based object detection model.");
    println!("This test verifies that burn-import can handle:");
    println!("  - Multi-head attention layers");
    println!("  - Complex transformer architectures");
    println!("  - Deformable attention mechanisms");
    println!();
    println!("Related issue: https://github.com/tracel-ai/burn/issues/4052");
}
