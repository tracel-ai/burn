//! Pipeline utilities for ONNX model validation

use crate::protos::OperatorSetIdProto;

/// Check whether the provided operator set version is supported
pub fn check_opset_version(opset: &OperatorSetIdProto, min_version: usize) -> bool {
    match opset.domain.as_str() {
        // Standard ONNX operators
        "" => opset.version >= min_version as i64,
        // ONNX ML operators - commonly used for traditional ML operators
        "ai.onnx.ml" => opset.version >= 1, // ML operators are generally stable from version 1
        // Add support for other domains as needed
        _ => {
            panic!(
                "Unsupported ONNX domain: '{}'. Only standard ONNX ('') and ML ('ai.onnx.ml') domains are supported",
                opset.domain
            );
        }
    }
}

/// Verify that all operator sets in a model are supported
pub fn verify_opsets(opsets: &[OperatorSetIdProto], min_version: usize) -> bool {
    for opset in opsets {
        if !check_opset_version(opset, min_version) {
            return false;
        }
    }
    true
}
