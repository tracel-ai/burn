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
    onnx_ir::parse_onnx(&model_path)
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
        .filter(|n| n.node_type == node_type)
        .count()
}

/// Count Constant nodes in the graph
pub fn count_constant_nodes(graph: &onnx_ir::ir::OnnxGraph) -> usize {
    count_nodes(graph, onnx_ir::ir::NodeType::Constant)
}

/// Count operation nodes (non-Constant nodes) in the graph
pub fn count_operation_nodes(graph: &onnx_ir::ir::OnnxGraph) -> usize {
    graph
        .nodes
        .iter()
        .filter(|n| !matches!(n.node_type, onnx_ir::ir::NodeType::Constant))
        .count()
}

/// Check if a graph contains a specific node type
pub fn has_node_type(graph: &onnx_ir::ir::OnnxGraph, node_type: onnx_ir::ir::NodeType) -> bool {
    graph.nodes.iter().any(|n| n.node_type == node_type)
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
