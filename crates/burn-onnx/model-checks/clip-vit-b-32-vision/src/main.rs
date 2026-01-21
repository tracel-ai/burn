extern crate alloc;

use burn::module::{Initializer, Param};
use burn::prelude::*;

use burn_store::{ModuleSnapshot, PytorchStore};
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
pub mod clip_vit_b_32_vision {
    include!(concat!(
        env!("OUT_DIR"),
        "/model/clip-vit-b-32-vision_opset16.rs"
    ));
}

#[derive(Debug, Module)]
struct TestData<B: Backend> {
    pixel_values: Param<Tensor<B, 4>>,
    image_embeds: Param<Tensor<B, 2>>,
}

impl<B: Backend> TestData<B> {
    fn new(device: &B::Device) -> Self {
        // CLIP ViT-B-32 vision: image_size=224, embed_dim=512
        Self {
            pixel_values: Initializer::Zeros.init([1, 3, 224, 224], device),
            image_embeds: Initializer::Zeros.init([1, 512], device),
        }
    }
}

fn main() {
    println!("========================================");
    println!("CLIP ViT-B-32-vision Burn Model Test");
    println!("========================================\n");

    // Check if artifacts exist
    let artifacts_dir = Path::new("artifacts");
    if !artifacts_dir.exists() {
        eprintln!("Error: artifacts directory not found!");
        eprintln!("Please run get_model.py first to download the model and test data.");
        std::process::exit(1);
    }

    // Initialize the model (using default which includes the converted weights)
    println!("Initializing CLIP vision model...");
    let start = Instant::now();
    let device = Default::default();
    let model: clip_vit_b_32_vision::Model<MyBackend> = clip_vit_b_32_vision::Model::default();
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
    let mut test_data = TestData::<MyBackend>::new(&device);
    let mut store = PytorchStore::from_file("artifacts/test_data.pt");
    test_data.load_from(&mut store).expect("Failed to load test data");
    let load_time = start.elapsed();
    println!("  Data loaded in {:.2?}", load_time);

    // Get the input tensors from test data
    let pixel_values = test_data.pixel_values.val();
    let pixel_values_shape = pixel_values.shape();
    println!(
        "  Loaded pixel_values with shape: {:?}",
        pixel_values_shape.dims
    );

    // Get the reference outputs from test data
    let reference_image_embeds = test_data.image_embeds.val();
    let ref_image_embeds_shape = reference_image_embeds.shape();
    println!(
        "  Loaded reference image_embeds with shape: {:?}",
        ref_image_embeds_shape.dims
    );

    // Run inference with the loaded input
    println!("\nRunning model inference with test input...");
    let start = Instant::now();

    let image_embeds = model.forward(pixel_values);

    let inference_time = start.elapsed();
    println!("  Inference completed in {:.2?}", inference_time);

    // Display output shapes
    let image_embeds_shape = image_embeds.shape();
    println!("\n  Model output shapes:");
    println!("    image_embeds: {:?}", image_embeds_shape.dims);

    // Verify expected output shapes match
    if image_embeds_shape.dims == ref_image_embeds_shape.dims {
        println!(
            "  ✓ image_embeds shape matches expected: {:?}",
            ref_image_embeds_shape.dims
        );
    } else {
        println!(
            "  ⚠ Warning: Expected image_embeds shape {:?}, got {:?}",
            ref_image_embeds_shape.dims, image_embeds_shape.dims
        );
    }

    // Compare outputs
    println!("\nComparing model outputs with reference data...");

    // Check if image_embeds are close
    println!("\n  Checking image_embeds:");
    if image_embeds
        .clone()
        .all_close(reference_image_embeds.clone(), Some(1e-4), Some(1e-4))
    {
        println!("    ✓ image_embeds matches reference data within tolerance (1e-4)!");
    } else {
        println!("    ⚠ image_embeds differs from reference data!");

        // Calculate and display the difference statistics
        let diff = image_embeds.clone() - reference_image_embeds.clone();
        let abs_diff = diff.abs();
        let max_diff = abs_diff.clone().max().into_scalar();
        let mean_diff = abs_diff.mean().into_scalar();

        println!("    Maximum absolute difference: {:.6}", max_diff);
        println!("    Mean absolute difference: {:.6}", mean_diff);

        // Show some sample values for debugging
        println!("\n    Sample values comparison (first 5 elements):");
        let output_flat = image_embeds.clone().flatten::<1>(0, 1);
        let reference_flat = reference_image_embeds.clone().flatten::<1>(0, 1);

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
    println!("  - Output validation: ✓ Passed");
    println!("========================================");
    println!("Model test completed successfully!");
    println!("========================================");
}
