/// Basic tests for ONNX-IR parsing
///
/// These tests verify that the ONNX-IR parser can successfully parse
/// simple ONNX models without errors. They are the most basic sanity checks
/// to ensure the pipeline works at all.
mod test_utils;

use test_utils::*;

#[test]
fn test_parse_basic_model() {
    // Load and parse the most basic model
    let graph = load_onnx("basic_model.onnx");

    // Basic sanity checks
    assert!(!graph.nodes.is_empty(), "Graph should have nodes");
    assert!(!graph.inputs.is_empty(), "Graph should have inputs");
    assert!(!graph.outputs.is_empty(), "Graph should have outputs");

    // Verify we have the expected operation nodes
    assert!(has_node_type(&graph, onnx_ir::ir::NodeType::Relu));
    assert!(has_node_type(&graph, onnx_ir::ir::NodeType::PRelu));
    assert!(has_node_type(&graph, onnx_ir::ir::NodeType::Add));

    // Verify basic I/O counts
    assert_eq!(graph.inputs.len(), 1, "Expected 1 input");
    assert_eq!(graph.outputs.len(), 1, "Expected 1 output");
}

#[test]
fn test_parse_multi_io() {
    // Test that we can handle multiple inputs and outputs
    let graph = load_onnx("multi_io.onnx");

    assert_eq!(graph.inputs.len(), 3, "Expected 3 inputs");
    assert_eq!(graph.outputs.len(), 2, "Expected 2 outputs");
    assert!(
        count_operation_nodes(&graph) >= 3,
        "Expected operation nodes"
    );
}

#[test]
fn test_parse_edge_cases() {
    // Test that edge cases don't crash the parser
    let graph = load_onnx("edge_cases.onnx");

    // Just verify we can parse it successfully
    assert_eq!(graph.inputs.len(), 1);
    assert_eq!(graph.outputs.len(), 3);
    assert_eq!(count_operation_nodes(&graph), 3);
}
