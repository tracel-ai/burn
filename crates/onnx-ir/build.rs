use protobuf_codegen::Customize;

fn main() {
    // Generate the onnx protobuf files
    // Enable bytes::Bytes generation for protobuf bytes fields instead of Vec<u8>
    // This enables zero-copy parsing when combined with mmap
    protobuf_codegen::Codegen::new()
        .pure()
        .includes(["src"])
        .input("src/protos/onnx.proto")
        .cargo_out_dir("onnx-protos")
        .customize(Customize::default().tokio_bytes(true))
        .run_from_script();
}
