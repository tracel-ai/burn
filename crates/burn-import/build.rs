fn main() {
    if cfg!(feature = "onnx") {
        // Generate the onnx protobuf files
        protobuf_codegen::Codegen::new()
            .pure()
            .includes(["src"])
            .input("src/onnx/protos/onnx.proto")
            .cargo_out_dir("onnx-protos")
            .run_from_script();
    }
}
