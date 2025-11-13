#![allow(dead_code)]

/// Test utilities for ONNX-IR integration tests
///
/// Provides helper functions for loading ONNX models and common test assertions.
use std::path::PathBuf;

/// Load an ONNX model from the tests directory
///
/// # Arguments
/// * `model_name` - Name of the ONNX file (e.g., "basic_model.onnx")
///
/// # Returns
/// Parsed OnnxGraph
///
/// # Panics
/// Panics if the model file doesn't exist or parsing fails
pub fn load_onnx(model_name: &str) -> onnx_ir::ir::OnnxGraph {
    let model_path = get_model_path(model_name);
    onnx_ir::parse_onnx(&model_path).unwrap_or_else(|e| {
        panic!(
            "Failed to parse ONNX model '{}': {}",
            model_path.display(),
            e
        )
    })
}

/// Get the path to an ONNX model in the tests/fixtures directory
///
/// # Arguments
/// * `model_name` - Name of the ONNX file
///
/// # Returns
/// PathBuf to the model file
pub fn get_model_path(model_name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(model_name)
}

/// Helper to check if a Node matches a NodeType
///
/// This uses a simple string comparison of variant names since both Node and NodeType
/// use the same naming convention. This is much simpler than manually listing all variants.
fn node_matches_type(node: &onnx_ir::ir::Node, node_type: onnx_ir::ir::NodeType) -> bool {
    // Get the variant name from the Debug representation
    // Node formats as "NodeVariant { ... }" and NodeType formats as "NodeVariant"
    let node_debug = format!("{:?}", node);
    let node_type_debug = format!("{:?}", node_type);

    // Extract variant name (everything before '{' or end of string)
    let node_variant = node_debug.split('{').next().unwrap().trim();

    node_variant == node_type_debug
}

/// Count nodes of a specific type in the graph
///
/// # Arguments
/// * `graph` - The OnnxGraph to search
/// * `node_type` - The NodeType to count
///
/// # Returns
/// Number of nodes matching the type
pub fn count_nodes(graph: &onnx_ir::ir::OnnxGraph, node_type: onnx_ir::ir::NodeType) -> usize {
    graph
        .nodes
        .iter()
        .filter(|n| node_matches_type(n, node_type.clone()))
        .count()
}

/// Count Constant nodes in the graph
pub fn count_constant_nodes(graph: &onnx_ir::ir::OnnxGraph) -> usize {
    graph
        .nodes
        .iter()
        .filter(|n| matches!(n, onnx_ir::ir::Node::Constant { .. }))
        .count()
}

/// Count operation nodes (non-Constant nodes) in the graph
pub fn count_operation_nodes(graph: &onnx_ir::ir::OnnxGraph) -> usize {
    graph
        .nodes
        .iter()
        .filter(|n| !matches!(n, onnx_ir::ir::Node::Constant { .. }))
        .count()
}

/// Check if a graph contains a specific node type
pub fn has_node_type(graph: &onnx_ir::ir::OnnxGraph, node_type: onnx_ir::ir::NodeType) -> bool {
    graph
        .nodes
        .iter()
        .any(|n| node_matches_type(n, node_type.clone()))
}

/// Get all unique data types from graph inputs
pub fn get_input_dtypes(graph: &onnx_ir::ir::OnnxGraph) -> Vec<burn_tensor::DType> {
    graph.inputs.iter().map(|inp| inp.ty.elem_type()).collect()
}

/// Count outputs by ArgType variant
pub fn count_outputs_by_type(graph: &onnx_ir::ir::OnnxGraph) -> (usize, usize, usize) {
    use onnx_ir::ir::ArgType;

    let scalars = graph
        .outputs
        .iter()
        .filter(|out| matches!(out.ty, ArgType::Scalar(_)))
        .count();

    let shapes = graph
        .outputs
        .iter()
        .filter(|out| matches!(out.ty, ArgType::Shape(_)))
        .count();

    let tensors = graph
        .outputs
        .iter()
        .filter(|out| matches!(out.ty, ArgType::Tensor(_)))
        .count();

    (scalars, shapes, tensors)
}
