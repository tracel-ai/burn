use burn_import::onnx::ModelGen;

fn main() {
    // Add onnx models.
    ModelGen::new()
        .input("tests/add/add.onnx")
        .out_dir("model/")
        .run_from_script();

    // panic!("Purposefully failing build to output logs.");
}
