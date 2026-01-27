//! LPIPS pretrained weights test example.
//!
//! This example loads pretrained LPIPS weights from PyTorch and compares
//! the output with PyTorch's reference values using various test images.
//!
//! ## Prerequisites
//!
//! Run the Python scripts first:
//! ```bash
//! cd crates/burn-nn/scripts
//! python save_lpips_weights.py      # Generate lpips_vgg.pt
//! python test_with_images.py        # Generate test images
//! ```

use burn::backend::NdArray;
use burn::nn::loss::{lpips_key_remaps, LpipsConfig, LpipsNet, Reduction};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, Recorder};
use burn::tensor::{TensorData, Tolerance};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};
use std::fs::File;
use std::io::Read;

type Backend = NdArray<f32>;
type FT = f32;

const WEIGHTS_PATH: &str = "crates/burn-nn/scripts/lpips_vgg.pt";
const SCRIPTS_PATH: &str = "crates/burn-nn/scripts";

/// Load raw float32 tensor from file
fn load_raw_image(path: &str, shape: [usize; 4], device: &<Backend as burn::tensor::backend::Backend>::Device) -> Tensor<Backend, 4> {
    let mut file = File::open(path).expect(&format!("Failed to open {}", path));
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).unwrap();

    let floats: Vec<f32> = buffer
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let data = TensorData::new(floats, shape);
    Tensor::from_data(data, device)
}

fn main() {
    println!("=== LPIPS Pretrained Weights Test ===\n");

    let device = Default::default();

    // Build LoadArgs with key remappings
    let key_remaps = lpips_key_remaps(LpipsNet::Vgg);
    let mut load_args = LoadArgs::new(WEIGHTS_PATH.into());
    for (pattern, replacement) in key_remaps {
        load_args = load_args.with_key_remap(pattern, replacement);
    }

    println!("Loading weights from: {}", WEIGHTS_PATH);
    println!("Applying {} key remappings...\n", key_remaps.len());

    // Load weights
    let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
        .load(load_args, &device)
        .expect("Failed to load LPIPS pretrained weights");

    // Create model with pretrained weights
    let lpips = LpipsConfig::new()
        .with_normalize(true)
        .init::<Backend>(&device)
        .load_record(record);

    println!("Model loaded successfully!\n");

    // =========================================================================
    // Test 1: zeros vs ones (32x32)
    // =========================================================================
    println!("--- Test 1: zeros vs ones (32x32) ---");
    let img1 = Tensor::<Backend, 4>::zeros([1, 3, 32, 32], &device);
    let img2 = Tensor::<Backend, 4>::ones([1, 3, 32, 32], &device);

    let loss = lpips.forward(img1, img2, Reduction::Mean);
    let loss_value = loss.to_data().to_vec::<f32>().unwrap()[0];

    // PyTorch: loss_fn(zeros, ones, normalize=True) = 0.511287
    let expected = TensorData::from([0.511287_f32]);
    println!("Burn:    {:.6}", loss_value);
    println!("PyTorch: {:.6}", 0.511287);

    loss.into_data()
        .assert_approx_eq::<FT>(&expected, Tolerance::default());
    println!("✓ PASSED\n");

    // =========================================================================
    // Test 2: Gradient images (64x64)
    // =========================================================================
    let img_h_path = format!("{}/test_img_horizontal.raw", SCRIPTS_PATH);
    let img_v_path = format!("{}/test_img_vertical.raw", SCRIPTS_PATH);
    let img_d_path = format!("{}/test_img_diagonal.raw", SCRIPTS_PATH);

    // Check if gradient images exist
    if std::path::Path::new(&img_h_path).exists() {
        println!("--- Test 2: Gradient images (64x64) ---");

        let shape = [1, 3, 64, 64];
        let img_h = load_raw_image(&img_h_path, shape, &device);
        let img_v = load_raw_image(&img_v_path, shape, &device);
        let img_d = load_raw_image(&img_d_path, shape, &device);

        // Test horizontal vs vertical
        // PyTorch: 0.686618
        let loss = lpips.forward(img_h.clone(), img_v.clone(), Reduction::Mean);
        let loss_value = loss.to_data().to_vec::<f32>().unwrap()[0];
        let expected = TensorData::from([0.686618_f32]);
        println!("horizontal vs vertical: Burn={:.6}, PyTorch={:.6}", loss_value, 0.686618);
        loss.into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
        println!("✓ PASSED");

        // Test horizontal vs diagonal
        // PyTorch: 0.693684
        let loss = lpips.forward(img_h.clone(), img_d.clone(), Reduction::Mean);
        let loss_value = loss.to_data().to_vec::<f32>().unwrap()[0];
        let expected = TensorData::from([0.693684_f32]);
        println!("horizontal vs diagonal: Burn={:.6}, PyTorch={:.6}", loss_value, 0.693684);
        loss.into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
        println!("✓ PASSED");

        // Test vertical vs diagonal
        // PyTorch: 0.557329
        let loss = lpips.forward(img_v.clone(), img_d.clone(), Reduction::Mean);
        let loss_value = loss.to_data().to_vec::<f32>().unwrap()[0];
        let expected = TensorData::from([0.557329_f32]);
        println!("vertical vs diagonal:   Burn={:.6}, PyTorch={:.6}", loss_value, 0.557329);
        loss.into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
        println!("✓ PASSED");

        // Test same image (should be 0)
        let loss = lpips.forward(img_h.clone(), img_h, Reduction::Mean);
        let loss_value = loss.to_data().to_vec::<f32>().unwrap()[0];
        let expected = TensorData::from([0.0_f32]);
        println!("horizontal vs horizontal: Burn={:.6}, PyTorch={:.6}", loss_value, 0.0);
        loss.into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
        println!("✓ PASSED\n");
    } else {
        println!("--- Test 2: Skipped (gradient images not found) ---");
        println!("Run: cd crates/burn-nn/scripts && python test_with_images.py\n");
    }

    println!("=== All tests PASSED! ===");
}
