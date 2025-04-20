//  Generated model from squeezenet1.onnx
mod internal_model {
    include!(concat!(env!("OUT_DIR"), "/model/squeezenet1_opset16.rs"));
}

pub use internal_model::*;
