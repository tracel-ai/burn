mod inner {
    include!(concat!(env!("OUT_DIR"), "/onnx-protos/mod.rs"));
}

pub use inner::onnx::*;
