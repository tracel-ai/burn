/// Smoke tests for ONNX-IR parsing
///
/// These tests verify that the ONNX-IR parser can successfully parse
/// basic ONNX models without errors.

use std::path::PathBuf;

#[test]
fn test_parse_basic_model() {
    // Path to the generated ONNX model
    let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("basic_model.onnx");

    // Parse the model - this will panic if parsing fails
    let graph = onnx_ir::parse_onnx(&model_path);

    // Basic sanity checks
    assert!(!graph.nodes.is_empty(), "Graph should have nodes");
    assert!(!graph.inputs.is_empty(), "Graph should have inputs");
    assert!(!graph.outputs.is_empty(), "Graph should have outputs");

    // Verify we have at least the operation nodes
    // (there may be additional nodes for constants/initializers)
    assert!(
        graph.nodes.len() >= 3,
        "Expected at least 3 nodes (Relu, PRelu, Add)"
    );

    // Verify node types - check that key operation nodes are present
    let node_types: Vec<_> = graph.nodes.iter().map(|n| n.node_type.clone()).collect();
    assert!(
        node_types.contains(&onnx_ir::ir::NodeType::Relu),
        "Should contain Relu node"
    );
    assert!(
        node_types.contains(&onnx_ir::ir::NodeType::PRelu),
        "Should contain PRelu node"
    );
    assert!(
        node_types.contains(&onnx_ir::ir::NodeType::Add),
        "Should contain Add node"
    );

    // Print node info for debugging
    println!("Graph has {} nodes:", graph.nodes.len());
    for node in &graph.nodes {
        println!("  - {:?} (name: {:?})", node.node_type, node.name);
    }

    // Verify inputs
    assert_eq!(graph.inputs.len(), 1, "Expected 1 input");
    // Input name may be renamed during processing (e.g., "input" -> "input1")
    assert!(
        graph.inputs[0].name.starts_with("input"),
        "Input name should start with 'input', got: {}",
        graph.inputs[0].name
    );

    // Verify outputs
    assert_eq!(graph.outputs.len(), 1, "Expected 1 output");
    // Output name may be renamed during processing
    assert!(
        graph.outputs[0].name.starts_with("output")
            || graph.outputs[0].name.starts_with("add"),
        "Output name should be meaningful, got: {}",
        graph.outputs[0].name
    );
}
