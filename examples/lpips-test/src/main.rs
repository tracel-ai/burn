//! LPIPS pretrained weights test example.
//!
//! This example loads pretrained LPIPS weights from PyTorch and computes
//! LPIPS values for comparison with PyTorch.
//!
//! ## Prerequisites
//!
//! Generate weights file first (Python):
//! ```bash
//! cd examples/lpips-test
//! python compare.py --generate-weights
//! ```
//!
//! ## Run
//! ```bash
//! # Burn (run from project root)
//! cargo run -p lpips-test --release
//!
//! # PyTorch (for comparison)
//! cd examples/lpips-test && python compare.py
//! ```

use burn::backend::NdArray;
use burn::nn::loss::{lpips_key_remaps, Lpips, LpipsConfig, LpipsNet, Reduction};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, Recorder};
use burn::tensor::TensorData;
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};
use std::fs::File;
use std::io::Read;
use std::path::Path;

type Backend = NdArray<f32>;

/// Load a raw image file (64x64x3 f32 format) as a tensor
fn load_raw_image(path: &str, device: &<Backend as burn::tensor::backend::Backend>::Device) -> Tensor<Backend, 4> {
    let mut file = File::open(path).expect("Failed to open image file");
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).expect("Failed to read image file");

    // Convert bytes to f32 (little-endian)
    let floats: Vec<f32> = buffer
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    // Shape: [H, W, C] -> [1, C, H, W]
    let data = TensorData::new(floats, [64, 64, 3]);
    Tensor::<Backend, 3>::from_data(data, device)
        .permute([2, 0, 1])  // [C, H, W]
        .unsqueeze::<4>()    // [1, C, H, W]
}

/// Load LPIPS model with pretrained weights
fn load_lpips(weights_path: &str, net: LpipsNet, device: &<Backend as burn::tensor::backend::Backend>::Device) -> Option<Lpips<Backend>> {
    if !Path::new(weights_path).exists() {
        println!("Weights not found: {}", weights_path);
        println!("Run: cd examples/lpips-test && python compare.py --generate-weights");
        return None;
    }

    let key_remaps = lpips_key_remaps(net);
    let mut load_args = LoadArgs::new(weights_path.into());
    for (pattern, replacement) in key_remaps {
        load_args = load_args.with_key_remap(pattern, replacement);
    }

    let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
        .load(load_args, device)
        .expect("Failed to load weights");

    Some(
        LpipsConfig::new()
            .with_net(net)
            .with_normalize(true)
            .init::<Backend>(device)
            .load_record(record)
    )
}

fn run_tests(lpips: &Lpips<Backend>, device: &<Backend as burn::tensor::backend::Backend>::Device) {
    let img_h = "examples/lpips-test/img/test_img_horizontal.raw";
    let img_v = "examples/lpips-test/img/test_img_vertical.raw";
    let img_d = "examples/lpips-test/img/test_img_diagonal.raw";

    // Test 1: zeros vs ones
    let zeros = Tensor::<Backend, 4>::zeros([1, 3, 64, 64], device);
    let ones = Tensor::<Backend, 4>::ones([1, 3, 64, 64], device);
    let loss1 = lpips.forward(zeros, ones, Reduction::Mean);
    println!("  zeros vs ones:      {:.6}", loss1.to_data().to_vec::<f32>().unwrap()[0]);

    // Test 2: horizontal vs vertical
    let h = load_raw_image(img_h, device);
    let v = load_raw_image(img_v, device);
    let loss2 = lpips.forward(h, v, Reduction::Mean);
    println!("  horizontal vs vertical: {:.6}", loss2.to_data().to_vec::<f32>().unwrap()[0]);

    // Test 3: horizontal vs diagonal
    let h = load_raw_image(img_h, device);
    let d = load_raw_image(img_d, device);
    let loss3 = lpips.forward(h, d, Reduction::Mean);
    println!("  horizontal vs diagonal: {:.6}", loss3.to_data().to_vec::<f32>().unwrap()[0]);
}

fn main() {
    println!("=== LPIPS Test (Burn) ===\n");

    let device = Default::default();

    // VGG
    println!("[VGG]");
    if let Some(lpips_vgg) = load_lpips("examples/lpips-test/weights/lpips_vgg.pt", LpipsNet::Vgg, &device) {
        run_tests(&lpips_vgg, &device);
    }

    // AlexNet
    println!("\n[AlexNet]");
    if let Some(lpips_alex) = load_lpips("examples/lpips-test/weights/lpips_alex.pt", LpipsNet::Alex, &device) {
        run_tests(&lpips_alex, &device);
    }

    // SqueezeNet
    println!("\n[SqueezeNet]");
    if let Some(lpips_squeeze) = load_lpips("examples/lpips-test/weights/lpips_squeeze.pt", LpipsNet::Squeeze, &device) {
        run_tests(&lpips_squeeze, &device);
    }

    println!("\nCompare with: cd examples/lpips-test && python compare.py");
}
