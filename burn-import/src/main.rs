use burn_import::onnx::ModelGen;

/// Takes an ONNX file and generates a model from it
fn main() {
    let onnx_file = std::env::args().nth(1).expect("No input file provided");
    let output_dir = std::env::args()
        .nth(2)
        .expect("No output directory provided");

    // Generate the model code from the ONNX file.
    ModelGen::new()
        .input(onnx_file.as_str())
        .development(true)
        .out_dir(output_dir.as_str())
        .run_from_cli();
}
