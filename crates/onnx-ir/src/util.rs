use crate::ir::{ArgType, Node};
use crate::protos::OperatorSetIdProto;

pub fn shape_config(curr: &Node) -> (usize, usize) {
    if curr.inputs.len() != 1 {
        panic!(
            "Shape: multiple inputs are not supported (got {:?})",
            curr.inputs.len()
        );
    }

    // Extract the shape of the input tensor
    let tensor = match curr.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    // Default: all axes up to the last one (included)
    let mut start_dim: i64 = 0;
    let mut end_dim: i64 = tensor.rank as i64;

    // Extract the attributes
    for (key, value) in curr.attrs.iter() {
        match key.as_str() {
            "start" => start_dim = value.clone().into_i64(),
            "end" => end_dim = value.clone().into_i64(),
            _ => {}
        }
    }

    // If dim is negative, it is counted from the end
    if start_dim < 0 {
        start_dim += tensor.rank as i64;
    }
    if end_dim < 0 {
        end_dim += tensor.rank as i64;
    }

    (start_dim as usize, end_dim as usize)
}

/// Check whether the provided operator set version is supported.
///
/// # Arguments
///
/// * `opset` - The operator set to check
/// * `min_version` - The minimum supported version
///
/// # Returns
///
/// * `bool` - True if the opset version is supported, false otherwise
///
/// # Panics
///
/// * If the domain is not the empty ONNX domain
pub fn check_opset_version(opset: &OperatorSetIdProto, min_version: i64) -> bool {
    // For now, only empty domain (standard ONNX operators) is supported
    if !opset.domain.is_empty() {
        panic!("Only the standard ONNX domain is supported");
    }

    // Return true if the opset version is greater than or equal to min_version
    opset.version >= min_version
}

/// Verify that all operator sets in a model are supported.
///
/// # Arguments
///
/// * `opsets` - The operator sets to check
/// * `min_version` - The minimum supported version
///
/// # Returns
///
/// * `bool` - True if all opset versions are supported, false otherwise
pub fn verify_opsets(opsets: &[OperatorSetIdProto], min_version: i64) -> bool {
    for opset in opsets {
        if !check_opset_version(opset, min_version) {
            return false;
        }
    }
    true
}
