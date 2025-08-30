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

// Import the generated model code as a module
pub mod clip_vit_b_32_text {
    include!(concat!(
        env!("OUT_DIR"),
        "/model/clip-vit-b-32-text_opset16.rs"
    ));
}

#[derive(Debug, Module)]
struct TestData<B: Backend> {
    input_ids: Param<Tensor<B, 2, Int>>,
    attention_mask: Param<Tensor<B, 2, Int>>,
    text_embeds: Param<Tensor<B, 2>>,
    last_hidden_state: Param<Tensor<B, 3>>,
}

fn main() {
    println!("========================================");
    println!("CLIP ViT-B-32-text Burn Model Test");
    println!("========================================\n");

    // Check if artifacts exist
    let artifacts_dir = Path::new("artifacts");
    if !artifacts_dir.exists() {
        eprintln!("Error: artifacts directory not found!");
        eprintln!("Please run get_model.py first to download the model and test data.");
        std::process::exit(1);
    }

    // Initialize the model (using default which includes the converted weights)
    println!("Initializing CLIP model...");
    let start = Instant::now();
    let device = Default::default();
    let model: clip_vit_b_32_text::Model<MyBackend> = clip_vit_b_32_text::Model::default();
    let init_time = start.elapsed();
    println!("  Model initialized in {:.2?}", init_time);

    // Save model structure to file
    println!("\nSaving model structure to artifacts/model.txt...");
    let model_str = format!("{}", model);
    std::fs::write("artifacts/model.txt", &model_str)
        .expect("Failed to write model structure to file");
    println!("  Model structure saved");

    // Load test data from PyTorch file
    println!("\nLoading test data from artifacts/test_data.pt...");
    let start = Instant::now();
    let test_data: TestDataRecord<MyBackend> = PyTorchFileRecorder::<FullPrecisionSettings>::new()
        .load("artifacts/test_data.pt".into(), &device)
        .expect("Failed to load test data");
    let load_time = start.elapsed();
    println!("  Data loaded in {:.2?}", load_time);

    // Get the input tensors from test data
    let input_ids = test_data.input_ids.val();
    let attention_mask = test_data.attention_mask.val();
    let input_ids_shape = input_ids.shape();
    let attention_mask_shape = attention_mask.shape();
    println!("  Loaded input_ids with shape: {:?}", input_ids_shape.dims);
    println!(
        "  Loaded attention_mask with shape: {:?}",
        attention_mask_shape.dims
    );

    // Get the reference outputs from test data
    let reference_text_embeds = test_data.text_embeds.val();
    let reference_last_hidden_state = test_data.last_hidden_state.val();
    let ref_text_embeds_shape = reference_text_embeds.shape();
    let ref_last_hidden_shape = reference_last_hidden_state.shape();
    println!(
        "  Loaded reference text_embeds with shape: {:?}",
        ref_text_embeds_shape.dims
    );
    println!(
        "  Loaded reference last_hidden_state with shape: {:?}",
        ref_last_hidden_shape.dims
    );

    // Run inference with the loaded input
    println!("\nRunning model inference with test input...");
    let start = Instant::now();

    let (text_embeds, last_hidden_state) = model.forward(input_ids, attention_mask);

    let inference_time = start.elapsed();
    println!("  Inference completed in {:.2?}", inference_time);

    // Display output shapes
    let text_embeds_shape = text_embeds.shape();
    let last_hidden_shape = last_hidden_state.shape();
    println!("\n  Model output shapes:");
    println!("    text_embeds: {:?}", text_embeds_shape.dims);
    println!("    last_hidden_state: {:?}", last_hidden_shape.dims);

    // Verify expected output shapes match
    if text_embeds_shape.dims == ref_text_embeds_shape.dims {
        println!(
            "  ✓ text_embeds shape matches expected: {:?}",
            ref_text_embeds_shape.dims
        );
    } else {
        println!(
            "  ⚠ Warning: Expected text_embeds shape {:?}, got {:?}",
            ref_text_embeds_shape.dims, text_embeds_shape.dims
        );
    }

    if last_hidden_shape.dims == ref_last_hidden_shape.dims {
        println!(
            "  ✓ last_hidden_state shape matches expected: {:?}",
            ref_last_hidden_shape.dims
        );
    } else {
        println!(
            "  ⚠ Warning: Expected last_hidden_state shape {:?}, got {:?}",
            ref_last_hidden_shape.dims, last_hidden_shape.dims
        );
    }

    // Compare outputs
    println!("\nComparing model outputs with reference data...");

    // Check if text_embeds are close
    println!("\n  Checking text_embeds:");
    if text_embeds
        .clone()
        .all_close(reference_text_embeds.clone(), Some(1e-4), Some(1e-4))
    {
        println!("    ✓ text_embeds matches reference data within tolerance (1e-4)!");
    } else {
        println!("    ⚠ text_embeds differs from reference data!");

        // Calculate and display the difference statistics
        let diff = text_embeds.clone() - reference_text_embeds.clone();
        let abs_diff = diff.abs();
        let max_diff = abs_diff.clone().max().into_scalar();
        let mean_diff = abs_diff.mean().into_scalar();

        println!("    Maximum absolute difference: {:.6}", max_diff);
        println!("    Mean absolute difference: {:.6}", mean_diff);

        // Show some sample values for debugging
        println!("\n    Sample values comparison (first 5 elements):");
        let output_flat = text_embeds.clone().flatten::<1>(0, 1);
        let reference_flat = reference_text_embeds.clone().flatten::<1>(0, 1);

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

    // Check if last_hidden_state is close
    println!("\n  Checking last_hidden_state:");
    if last_hidden_state.clone().all_close(
        reference_last_hidden_state.clone(),
        Some(1e-4),
        Some(1e-4),
    ) {
        println!("    ✓ last_hidden_state matches reference data within tolerance (1e-4)!");
    } else {
        println!("    ⚠ last_hidden_state differs from reference data!");

        // Calculate and display the difference statistics
        let diff = last_hidden_state.clone() - reference_last_hidden_state.clone();
        let abs_diff = diff.abs();
        let max_diff = abs_diff.clone().max().into_scalar();
        let mean_diff = abs_diff.mean().into_scalar();

        println!("    Maximum absolute difference: {:.6}", max_diff);
        println!("    Mean absolute difference: {:.6}", mean_diff);

        // Show some sample values for debugging
        println!("\n    Sample values comparison (first 5 elements):");
        let output_flat = last_hidden_state.clone().flatten::<1>(0, 2);
        let reference_flat = reference_last_hidden_state.clone().flatten::<1>(0, 2);

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
    println!("Model test completed!");
    println!("========================================");
}
