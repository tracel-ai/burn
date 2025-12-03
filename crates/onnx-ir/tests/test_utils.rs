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
    onnx_ir::OnnxGraphBuilder::new()
        .parse_file(&model_path)
        .unwrap_or_else(|e| {
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

/// Count nodes matching a predicate
///
/// # Arguments
/// * `graph` - The OnnxGraph to search
/// * `predicate` - Function to test each node
///
/// # Returns
/// Number of nodes matching the predicate
///
/// # Example
/// ```ignore
/// count_nodes(&graph, |n| matches!(n, Node::PRelu { .. }))
/// ```
pub fn count_nodes<F>(graph: &onnx_ir::ir::OnnxGraph, predicate: F) -> usize
where
    F: Fn(&onnx_ir::ir::Node) -> bool,
{
    graph.nodes.iter().filter(|n| predicate(n)).count()
}

/// Count Constant nodes in the graph
pub fn count_constant_nodes(graph: &onnx_ir::ir::OnnxGraph) -> usize {
    count_nodes(graph, |n| matches!(n, onnx_ir::ir::Node::Constant { .. }))
}

/// Count operation nodes (non-Constant nodes) in the graph
pub fn count_operation_nodes(graph: &onnx_ir::ir::OnnxGraph) -> usize {
    graph
        .nodes
        .iter()
        .filter(|n| !matches!(n, onnx_ir::ir::Node::Constant { .. }))
        .count()
}

/// Check if a graph contains nodes matching a predicate
///
/// # Example
/// ```ignore
/// has_node_type(&graph, |n| matches!(n, Node::MatMul { .. }))
/// ```
pub fn has_node_type<F>(graph: &onnx_ir::ir::OnnxGraph, predicate: F) -> bool
where
    F: Fn(&onnx_ir::ir::Node) -> bool,
{
    graph.nodes.iter().any(predicate)
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
