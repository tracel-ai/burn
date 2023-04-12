use burn_import::onnx::ModelCodeGen;

fn main() {
    // Generate the model code from the ONNX file.
    ModelCodeGen::new()
        .input("src/model/mnist.onnx")
        .out_dir("model/")
        .run_from_script();
}
