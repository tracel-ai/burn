extern crate alloc;

use burn::module::{Initializer, Param};
use burn::prelude::*;

use burn_store::PytorchStore;
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

impl<B: Backend> TestData<B> {
    fn new(device: &B::Device) -> Self {
        // RF-DETR Small: input 512x512, 300 queries, 4 bbox coords, 91 classes (COCO)
        Self {
            input: Initializer::Zeros.init([1, 3, 512, 512], device),
            output_dets: Initializer::Zeros.init([1, 300, 4], device),
            output_labels: Initializer::Zeros.init([1, 300, 91], device),
        }
    }
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

    // Initialize the model with weights
    println!("Initializing RF-DETR Small model...");
    let start = Instant::now();
    let device = Default::default();

    // The model weights are generated at build time and stored in the OUT_DIR
    // We need to load them from the embedded burnpack file
    let weights_path = concat!(env!("OUT_DIR"), "/model/rf_detr_small.bpk");
    let model: Model<MyBackend> = Model::from_file(weights_path, &device);
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
    let mut test_data = TestData::<MyBackend>::new(&device);
    let mut store = PytorchStore::from_file(&test_data_file);
    test_data.load_from(&mut store).expect("Failed to load test data");
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

    // Compare outputs
    println!("\nComparing model outputs with reference data...");

    let mut dets_passed = false;
    let mut labels_passed = false;

    // Check if dets are close
    println!("\n  Checking dets (bounding boxes):");
    if output_dets
        .clone()
        .all_close(reference_dets.clone(), Some(1e-4), Some(1e-4))
    {
        println!("    ✓ dets matches reference data within tolerance (1e-4)!");
        dets_passed = true;
    } else {
        println!("    ⚠ dets differs from reference data!");

        // Calculate and display the difference statistics
        let diff = output_dets.clone() - reference_dets.clone();
        let abs_diff = diff.abs();
        let max_diff = abs_diff.clone().max().into_scalar();
        let mean_diff = abs_diff.mean().into_scalar();

        println!("    Maximum absolute difference: {:.6}", max_diff);
        println!("    Mean absolute difference: {:.6}", mean_diff);

        // Show some sample values for debugging
        println!("\n    Sample values comparison (first 5 elements):");
        let output_flat = output_dets.clone().flatten::<1>(0, 2);
        let reference_flat = reference_dets.clone().flatten::<1>(0, 2);

        for i in 0..5.min(output_flat.dims()[0]) {
            let model_val: f32 = output_flat.clone().slice(s![i..i + 1]).into_scalar();
            let ref_val: f32 = reference_flat.clone().slice(s![i..i + 1]).into_scalar();
            println!(
                "      [{}] Model: {:.6}, Reference: {:.6}, Diff: {:.6}",
                i,
                model_val,
                ref_val,
                (model_val - ref_val).abs()
            );
        }
    }

    // Check if labels are close
    println!("\n  Checking labels (class scores):");
    if output_labels
        .clone()
        .all_close(reference_labels.clone(), Some(1e-4), Some(1e-4))
    {
        println!("    ✓ labels matches reference data within tolerance (1e-4)!");
        labels_passed = true;
    } else {
        println!("    ⚠ labels differs from reference data!");

        // Calculate and display the difference statistics
        let diff = output_labels.clone() - reference_labels.clone();
        let abs_diff = diff.abs();
        let max_diff = abs_diff.clone().max().into_scalar();
        let mean_diff = abs_diff.mean().into_scalar();

        println!("    Maximum absolute difference: {:.6}", max_diff);
        println!("    Mean absolute difference: {:.6}", mean_diff);

        // Show some sample values for debugging
        println!("\n    Sample values comparison (first 5 elements):");
        let output_flat = output_labels.clone().flatten::<1>(0, 2);
        let reference_flat = reference_labels.clone().flatten::<1>(0, 2);

        for i in 0..5.min(output_flat.dims()[0]) {
            let model_val: f32 = output_flat.clone().slice(s![i..i + 1]).into_scalar();
            let ref_val: f32 = reference_flat.clone().slice(s![i..i + 1]).into_scalar();
            println!(
                "      [{}] Model: {:.6}, Reference: {:.6}, Diff: {:.6}",
                i,
                model_val,
                ref_val,
                (model_val - ref_val).abs()
            );
        }
    }

    println!("\n========================================");
    println!("Summary:");
    println!("  - Model initialization: {:.2?}", init_time);
    println!("  - Data loading: {:.2?}", load_time);
    println!("  - Inference time: {:.2?}", inference_time);
    if dets_passed && labels_passed {
        println!("  - Output validation: ✓ All outputs match!");
    } else {
        println!(
            "  - Output validation: {} dets, {} labels",
            if dets_passed { "✓" } else { "✗" },
            if labels_passed { "✓" } else { "✗" }
        );
    }
    println!("========================================");
    if dets_passed && labels_passed {
        println!("Model test completed successfully!");
    } else {
        println!("Model test completed with differences.");
    }
    println!("========================================");
}
