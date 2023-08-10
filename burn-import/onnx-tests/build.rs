use burn_import::onnx::ModelGen;

fn main() {
    // Re-run this build script if the onnx-tests directory changes.
    println!("cargo:rerun-if-changed=tests");

    // Add onnx models.
    ModelGen::new()
        .input("tests/add/add.onnx")
        .input("tests/sub/sub.onnx")
        .input("tests/mul/mul.onnx")
        .input("tests/div/div.onnx")
        .input("tests/concat/concat.onnx")
        .input("tests/conv2d/conv2d.onnx")
        .input("tests/dropout/dropout.onnx")
        .input("tests/global_avr_pool/global_avr_pool.onnx")
        .input("tests/softmax/softmax.onnx")
        .input("tests/log_softmax/log_softmax.onnx")
        .input("tests/maxpool2d/maxpool2d.onnx")
        .out_dir("model/")
        .run_from_script();

    // panic!("Purposefully failing build to output logs.");
}
