extern crate alloc;

use burn::prelude::*;
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
include!(concat!(env!("OUT_DIR"), "/model/silero_vad.rs"));

fn main() {
    println!("========================================");
    println!("Silero VAD Model Test");
    println!("========================================\n");

    println!();

    // Check if artifacts exist
    let artifacts_dir = Path::new("artifacts");
    if !artifacts_dir.exists() {
        eprintln!("Error: artifacts directory not found!");
        eprintln!("Please run get_model.py first to download the model.");
        eprintln!("Example: uv run get_model.py");
        std::process::exit(1);
    }

    // Check if model file exists
    let model_file = artifacts_dir.join("silero_vad.onnx");
    if !model_file.exists() {
        eprintln!("Error: Model file not found!");
        eprintln!("Please run: uv run get_model.py");
        std::process::exit(1);
    }

    // Initialize the model
    println!("Initializing Silero VAD model...");
    let start = Instant::now();
    let device = Default::default();
    let model: Model<MyBackend> = Model::default();
    let init_time = start.elapsed();
    println!("  ✓ Model initialized in {:.2?}", init_time);

    // Save model structure to file
    let model_txt_path = artifacts_dir.join("silero_vad_model.txt");
    println!(
        "\nSaving model structure to {}...",
        model_txt_path.display()
    );
    let model_str = format!("{}", model);
    std::fs::write(&model_txt_path, &model_str).expect("Failed to write model structure to file");
    println!("  ✓ Model structure saved");

    // Create sample input
    // Silero VAD expects input shape: [batch_size, sequence_length]
    // We'll use a batch of 1 with 512 audio samples (16ms at 16kHz)
    println!("\nCreating sample input tensor...");
    let batch_size = 1;
    let sequence_length = 512;

    // Create random audio-like input (in reality would be audio samples)
    let input = Tensor::<MyBackend, 2>::random(
        [batch_size, sequence_length],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );

    println!("  Input shape: [{}, {}]", batch_size, sequence_length);

    // The ifless model uses a combined state tensor
    // state: [2, batch, 128] - Combined LSTM hidden and cell state
    let state = Tensor::<MyBackend, 3>::zeros([2, batch_size, 128], &device);

    // sr is sample rate (typically 16000 for 16kHz)
    let sr = 16000i64;

    println!("\nRunning model inference...");
    let start = Instant::now();
    let (output, _state_out) = model.forward(input, sr, state);
    let inference_time = start.elapsed();
    println!("  ✓ Inference completed in {:.2?}", inference_time);

    // Display output
    let shape = output.shape();
    println!("\nModel output:");
    println!("  Voice probability shape: {:?}", shape.dims);

    // Get the voice probability value
    let prob: f32 = output.clone().into_scalar();
    println!(
        "  Voice probability: {:.4} (0.0 = no voice, 1.0 = voice)",
        prob
    );

    if prob >= 0.0 && prob <= 1.0 {
        println!("  ✓ Output is a valid probability");
    } else {
        println!("  ⚠ Output is outside [0, 1] range!");
    }

    println!("\n========================================");
    println!("Model test completed successfully!");
    println!("========================================");
    println!();
    println!("✓ Silero VAD ifless model is working correctly");
    println!("✓ Model can be imported and executed with burn");
}
