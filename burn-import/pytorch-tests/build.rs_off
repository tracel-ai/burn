use burn_import::pytorch::{Converter, RecordType};

fn main() {
    // Re-run this build script if the onnx-tests directory changes.
    println!("cargo:rerun-if-changed=tests");

    Converter::new()
        .input("tests/batch_norm/batch_norm2d.pt")
        .input("tests/layer_norm/layer_norm.pt")
        .input("tests/group_norm/group_norm.pt")
        .input("tests/embedding/embedding.pt")
        .input("tests/conv1d/conv1d.pt")
        .input("tests/conv2d/conv2d.pt")
        .input("tests/conv_transpose1d/conv_transpose1d.pt")
        .input("tests/conv_transpose2d/conv_transpose2d.pt")
        .input("tests/buffer/buffer.pt")
        .input("tests/boolean/boolean.pt")
        .input("tests/integer/integer.pt")
        .out_dir("model/")
        .run_from_script();

    Converter::new()
        .input("tests/linear/linear.pt")
        .out_dir("model/labeled/")
        .guess_module_type(false)
        .linear_module("fc.*")
        .run_from_script();

    Converter::new()
        .input("tests/linear/linear_with_bias.pt")
        .out_dir("model/guessed/")
        .run_from_script();

    Converter::new()
        .input("tests/key_remap/key_remap.pt")
        .out_dir("model/")
        .key_remap("conv\\.(.*)", "$1") // Remove "conv" prefix from all keys
        .run_from_script();

    Converter::new()
        .input("tests/complex_nested/complex_nested.pt")
        .out_dir("model/full/")
        .record_type(RecordType::PrettyJson)
        .run_from_script();

    Converter::new()
        .input("tests/complex_nested/complex_nested.pt")
        .out_dir("model/full/")
        .record_type(RecordType::NamedMpk)
        .run_from_script();

    Converter::new()
        .input("tests/complex_nested/complex_nested.pt")
        .out_dir("model/full/")
        .record_type(RecordType::NamedMpkGz)
        .run_from_script();

    Converter::new()
        .input("tests/complex_nested/complex_nested.pt")
        .out_dir("model/half/")
        .half_precision(true)
        .record_type(RecordType::PrettyJson)
        .run_from_script();

    Converter::new()
        .input("tests/complex_nested/complex_nested.pt")
        .out_dir("model/half/")
        .half_precision(true)
        .record_type(RecordType::NamedMpk)
        .run_from_script();

    Converter::new()
        .input("tests/complex_nested/complex_nested.pt")
        .out_dir("model/half/")
        .half_precision(true)
        .record_type(RecordType::NamedMpkGz)
        .run_from_script();
}
