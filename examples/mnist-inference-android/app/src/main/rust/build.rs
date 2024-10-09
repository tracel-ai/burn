use burn_import::onnx::{ModelGen, RecordType};

fn main() {
    // If the embedded-model, then model is bundled into the binary.
    ModelGen::new()
        .input("src/model/mnist.onnx")
        .out_dir("model/")
        .record_type(RecordType::Bincode)
        .embed_states(true)
        .run_from_script();
}
