use burn_import::onnx::{ModelGen, RecordType};

fn main() {
    // Generate the model code from the ONNX file.

    if cfg!(feature = "embedded-model") {
        // If the embedded-model, then model is bundled into the binary.
        ModelGen::new()
            .input("src/model/mnist.onnx")
            .out_dir("model/")
            .record_type(RecordType::Bincode)
            .embed_states(true)
            .run_from_script();
    } else {
        // If not embedded-model, then model is loaded from the file system (default).
        ModelGen::new()
            .input("src/model/mnist.onnx")
            .out_dir("model/")
            .run_from_script();
    }
}
