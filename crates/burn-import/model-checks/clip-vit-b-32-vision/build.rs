use burn_import::onnx::ModelGen;
use std::path::Path;

fn main() {
    let onnx_path = "artifacts/clip-vit-b-32-vision_opset16.onnx";
    let test_data_path = "artifacts/test_data.pt";

    // Tell Cargo to only rebuild if these files change
    println!("cargo:rerun-if-changed={}", onnx_path);
    println!("cargo:rerun-if-changed={}", test_data_path);
    println!("cargo:rerun-if-changed=build.rs");

    // Check if the ONNX model file exists
    if !Path::new(onnx_path).exists() {
        eprintln!("Error: ONNX model file not found at '{}'", onnx_path);
        eprintln!();
        eprintln!("Please run the following command to download and prepare the model:");
        eprintln!("  python get_model.py");
        eprintln!();
        eprintln!("Or if you prefer using uv:");
        eprintln!("  uv run get_model.py");
        eprintln!();
        eprintln!(
            "This will download the CLIP ViT-B-32-vision model and convert it to ONNX format."
        );
        std::process::exit(1);
    }

    // Generate the model code from the ONNX file
    // Use double precision to handle large Int64 constants in CLIP
    ModelGen::new()
        .input(onnx_path)
        .out_dir("model/")
        .run_from_script();
}
