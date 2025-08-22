extern crate alloc;

use burn::module::Param;
use burn::prelude::*;
use burn::record::*;

use burn_import::pytorch::PyTorchFileRecorder;
use std::time::Instant;

#[cfg(feature = "wgpu")]
pub type MyBackend = burn::backend::Wgpu;

#[cfg(feature = "ndarray")]
pub type MyBackend = burn::backend::NdArray<f32>;

#[cfg(feature = "tch")]
pub type MyBackend = burn::backend::LibTorch<f32>;

#[cfg(feature = "metal")]
pub type MyBackend = burn::backend::Metal;

// Import the generated model code as a module
#[allow(clippy::type_complexity)]
pub mod yolo11x {
    include!(concat!(env!("OUT_DIR"), "/model/yolo11x_opset16.rs"));
}

#[derive(Debug, Module)]
struct TestData<B: Backend> {
    input: Param<Tensor<B, 4>>,
    output: Param<Tensor<B, 3>>,
}

fn main() {
    println!("========================================");
    println!("YOLO11x Burn Model Test");
    println!("========================================\n");

    // Initialize the model (without weights for now)
    println!("Initializing YOLO11x model...");
    let start = Instant::now();
    let device = Default::default();
    let model: yolo11x::Model<MyBackend> = yolo11x::Model::default();
    let init_time = start.elapsed();
    println!("  Model initialized in {:.2?}", init_time);

    // Load test data from PyTorch file
    println!("\nLoading test data from artifacts/test_data.pt...");
    let start = Instant::now();
    let test_data: TestDataRecord<MyBackend> = PyTorchFileRecorder::<FullPrecisionSettings>::new()
        .load("artifacts/test_data.pt".into(), &device)
        .expect("Failed to load test data");
    let load_time = start.elapsed();
    println!("  Data loaded in {:.2?}", load_time);

    // Get the input tensor from test data
    let input = test_data.input.val();
    let input_shape = input.shape();
    println!("  Loaded input tensor with shape: {:?}", input_shape.dims);

    // Get the reference output from test data
    let reference_output = test_data.output.val();
    let reference_shape = reference_output.shape();
    println!(
        "  Loaded reference output with shape: {:?}",
        reference_shape.dims
    );

    // Run inference with the loaded input
    println!("\nRunning model inference with test input...");
    let start = Instant::now();
    let output = model.forward(input);
    let inference_time = start.elapsed();
    println!("  Inference completed in {:.2?}", inference_time);

    // Display output shape
    let shape = output.shape();
    println!("\n  Model output shape: {:?}", shape.dims);

    // Verify expected output shape
    let expected_shape = [1, 84, 8400];
    if shape.dims == expected_shape {
        println!("  ✓ Output shape matches expected: {:?}", expected_shape);
    } else {
        println!(
            "  ⚠ Warning: Expected shape {:?}, got {:?}",
            expected_shape, shape.dims
        );
    }

    // Compare outputs
    println!("\nComparing model output with reference data...");

    // Check if outputs are close
    if output
        .clone()
        .all_close(reference_output.clone(), Some(1e-4), Some(1e-4))
    {
        println!("  ✓ Model output matches reference data within tolerance (1e-4)!");
    } else {
        println!("  ⚠ Model output differs from reference data!");

        // Calculate and display the difference statistics
        let diff = output.clone() - reference_output.clone();
        let abs_diff = diff.abs();
        let max_diff = abs_diff.clone().max().into_scalar();
        let mean_diff = abs_diff.mean().into_scalar();

        println!("  Maximum absolute difference: {:.6}", max_diff);
        println!("  Mean absolute difference: {:.6}", mean_diff);

        // Show some sample values for debugging
        println!("\n  Sample values comparison (first 5 elements):");
        let output_flat = output.clone().flatten::<1>(0, 2);
        let reference_flat = reference_output.clone().flatten::<1>(0, 2);

        for i in 0..5.min(output_flat.dims()[0]) {
            let model_val: f32 = output_flat.clone().slice([i..i + 1]).into_scalar();
            let ref_val: f32 = reference_flat.clone().slice([i..i + 1]).into_scalar();
            println!(
                "    [{}] Model: {:.6}, Reference: {:.6}, Diff: {:.6}",
                i,
                model_val,
                ref_val,
                (model_val - ref_val).abs()
            );
        }
    }

    println!("\n========================================");
    println!("Model test completed!");
    println!("========================================");
}
