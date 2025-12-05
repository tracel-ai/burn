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
pub mod all_minilm_l6_v2 {
    include!(concat!(
        env!("OUT_DIR"),
        "/model/all-minilm-l6-v2_opset16.rs"
    ));
}

#[derive(Debug, Module)]
struct TestData<B: Backend> {
    input_ids: Param<Tensor<B, 2, Int>>,
    attention_mask: Param<Tensor<B, 2, Int>>,
    token_type_ids: Param<Tensor<B, 2, Int>>,
    last_hidden_state: Param<Tensor<B, 3>>,
    pooled_embeddings: Param<Tensor<B, 2>>,
}

impl<B: Backend> TestData<B> {
    fn new(device: &B::Device) -> Self {
        use burn::module::ParamId;
        // Initialize with correct shapes matching the test data
        // Note: Initializer only works for float tensors, Int tensors need manual init
        Self {
            input_ids: Param::initialized(ParamId::new(), Tensor::zeros([1, 128], device)),
            attention_mask: Param::initialized(ParamId::new(), Tensor::zeros([1, 128], device)),
            token_type_ids: Param::initialized(ParamId::new(), Tensor::zeros([1, 128], device)),
            last_hidden_state: Initializer::Zeros.init([1, 128, 384], device),
            pooled_embeddings: Initializer::Zeros.init([1, 384], device),
        }
    }
}

/// Apply mean pooling to get sentence embeddings
fn mean_pool<B: Backend>(
    last_hidden_state: Tensor<B, 3>,
    attention_mask: Tensor<B, 2, Int>,
) -> Tensor<B, 2> {
    // Convert attention_mask to float and expand dimensions to match hidden_state
    let attention_mask_float = attention_mask.float().unsqueeze_dim::<3>(2);

    // Multiply hidden states by attention mask
    let masked_embeddings = last_hidden_state * attention_mask_float.clone();

    // Sum along sequence dimension (dim 1)
    let sum_embeddings = masked_embeddings.sum_dim(1);

    // Sum attention mask to get count of non-padding tokens
    let sum_mask = attention_mask_float.sum_dim(1).clamp_min(1e-9);

    // Divide to get mean - result is [batch, 1, hidden]
    let pooled = sum_embeddings / sum_mask;

    // Get the shape to reshape to [batch, hidden]
    let shape = pooled.shape();
    let batch_size = shape.dims[0];
    let hidden_size = shape.dims[2];

    pooled.reshape([batch_size, hidden_size])
}

fn main() {
    println!("========================================");
    println!("all-MiniLM-L6-v2 Burn Model Test");
    println!("========================================\n");

    // Check if artifacts exist
    let artifacts_dir = Path::new("artifacts");
    if !artifacts_dir.exists() {
        eprintln!("Error: artifacts directory not found!");
        eprintln!("Please run get_model.py first to download the model and test data.");
        std::process::exit(1);
    }

    // Initialize the model (using default which includes the converted weights)
    println!("Initializing all-MiniLM-L6-v2 model...");
    let start = Instant::now();
    let device = Default::default();
    let model: all_minilm_l6_v2::Model<MyBackend> = all_minilm_l6_v2::Model::default();
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
    let input_ids = test_data.input_ids.val();
    let attention_mask = test_data.attention_mask.val();
    let token_type_ids = test_data.token_type_ids.val();
    let input_ids_shape = input_ids.shape();
    let attention_mask_shape = attention_mask.shape();
    let token_type_ids_shape = token_type_ids.shape();
    println!("  Loaded input_ids with shape: {:?}", input_ids_shape.dims);
    println!(
        "  Loaded attention_mask with shape: {:?}",
        attention_mask_shape.dims
    );
    println!(
        "  Loaded token_type_ids with shape: {:?}",
        token_type_ids_shape.dims
    );

    // Get the reference outputs from test data
    let reference_last_hidden_state = test_data.last_hidden_state.val();
    let reference_pooled_embeddings = test_data.pooled_embeddings.val();
    let ref_last_hidden_shape = reference_last_hidden_state.shape();
    let ref_pooled_shape = reference_pooled_embeddings.shape();
    println!(
        "  Loaded reference last_hidden_state with shape: {:?}",
        ref_last_hidden_shape.dims
    );
    println!(
        "  Loaded reference pooled_embeddings with shape: {:?}",
        ref_pooled_shape.dims
    );

    // Run inference with the loaded input
    println!("\nRunning model inference with test input...");
    let start = Instant::now();

    let last_hidden_state = model.forward(
        input_ids.clone(),
        attention_mask.clone(),
        token_type_ids.clone(),
    );

    let inference_time = start.elapsed();
    println!("  Inference completed in {:.2?}", inference_time);

    // Compute pooled embeddings (mean pooling)
    println!("\nComputing pooled embeddings (mean pooling)...");
    let start = Instant::now();
    let pooled_embeddings = mean_pool(last_hidden_state.clone(), attention_mask.clone());
    let pooling_time = start.elapsed();
    println!("  Pooling completed in {:.2?}", pooling_time);

    // Display output shapes
    let last_hidden_shape = last_hidden_state.shape();
    let pooled_shape = pooled_embeddings.shape();
    println!("\n  Model output shapes:");
    println!("    last_hidden_state: {:?}", last_hidden_shape.dims);
    println!("    pooled_embeddings: {:?}", pooled_shape.dims);

    // Verify expected output shapes match
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

    if pooled_shape.dims == ref_pooled_shape.dims {
        println!(
            "  ✓ pooled_embeddings shape matches expected: {:?}",
            ref_pooled_shape.dims
        );
    } else {
        println!(
            "  ⚠ Warning: Expected pooled_embeddings shape {:?}, got {:?}",
            ref_pooled_shape.dims, pooled_shape.dims
        );
    }

    // Compare outputs
    println!("\nComparing model outputs with reference data...");

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

    // Check if pooled_embeddings is close
    println!("\n  Checking pooled_embeddings:");
    if pooled_embeddings.clone().all_close(
        reference_pooled_embeddings.clone(),
        Some(1e-4),
        Some(1e-4),
    ) {
        println!("    ✓ pooled_embeddings matches reference data within tolerance (1e-4)!");
    } else {
        println!("    ⚠ pooled_embeddings differs from reference data!");

        // Calculate and display the difference statistics
        let diff = pooled_embeddings.clone() - reference_pooled_embeddings.clone();
        let abs_diff = diff.abs();
        let max_diff = abs_diff.clone().max().into_scalar();
        let mean_diff = abs_diff.mean().into_scalar();

        println!("    Maximum absolute difference: {:.6}", max_diff);
        println!("    Mean absolute difference: {:.6}", mean_diff);

        // Show some sample values for debugging
        println!("\n    Sample values comparison (first 5 elements):");
        let output_flat = pooled_embeddings.clone().flatten::<1>(0, 1);
        let reference_flat = reference_pooled_embeddings.clone().flatten::<1>(0, 1);

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
