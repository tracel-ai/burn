use burn_import::onnx::ModelGen;
use std::env;
use std::fs;
use std::path::Path;

fn main() {
    // Supported models
    let supported_models = vec!["yolov5s", "yolov8n", "yolov8s", "yolov10n", "yolo11x"];

    // Get the model name from environment variable (required)
    let model_name = env::var("YOLO_MODEL").unwrap_or_else(|_| {
        eprintln!("Error: YOLO_MODEL environment variable is not set.");
        eprintln!();
        eprintln!("Please specify which YOLO model to build:");
        eprintln!("  YOLO_MODEL=yolov8n cargo build");
        eprintln!();
        eprintln!("Available models: {}", supported_models.join(", "));
        std::process::exit(1);
    });

    if !supported_models.contains(&model_name.as_str()) {
        eprintln!(
            "Error: Unsupported model '{}'. Supported models: {:?}",
            model_name, supported_models
        );
        std::process::exit(1);
    }

    let onnx_path = format!("artifacts/{}_opset16.onnx", model_name);
    let test_data_path = format!("artifacts/{}_test_data.pt", model_name);

    // Tell Cargo to only rebuild if these files change
    println!("cargo:rerun-if-changed={}", onnx_path);
    println!("cargo:rerun-if-changed={}", test_data_path);
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=YOLO_MODEL");

    // Check if the ONNX model file exists
    if !Path::new(&onnx_path).exists() {
        eprintln!("Error: ONNX model file not found at '{}'", onnx_path);
        eprintln!();
        eprintln!(
            "Please run the following command to download and prepare the {} model:",
            model_name
        );
        eprintln!("  python get_model.py --model {}", model_name);
        eprintln!();
        eprintln!("Or if you prefer using uv:");
        eprintln!("  uv run get_model.py --model {}", model_name);
        eprintln!();
        eprintln!("Available models: {}", supported_models.join(", "));
        std::process::exit(1);
    }

    // Generate the model code from the ONNX file
    ModelGen::new()
        .input(&onnx_path)
        .out_dir("model/")
        .run_from_script();

    // Write the model name to a file so main.rs can access it
    let out_dir = env::var("OUT_DIR").unwrap();
    let model_info_path = Path::new(&out_dir).join("model_info.rs");

    // Generate the include path for the model
    let model_include = format!(
        "include!(concat!(env!(\"OUT_DIR\"), \"/model/{}_opset16.rs\"));",
        model_name
    );

    fs::write(
        model_info_path,
        format!(
            r#"pub const MODEL_NAME: &str = "{}";
pub const TEST_DATA_FILE: &str = "{}_test_data.pt";

// Include the generated model
pub mod yolo_model {{
    {}
}}"#,
            model_name, model_name, model_include
        ),
    )
    .expect("Failed to write model info");
}
