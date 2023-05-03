use burn_import::onnx::ModelGen;

fn main() {
    // Generate the model code from the ONNX file.
    ModelGen::new()
        .input("src/model/mnist.onnx")
        .out_dir("model/")
        .run_from_script();
}
