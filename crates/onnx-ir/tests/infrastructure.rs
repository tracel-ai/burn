/// Infrastructure tests for ONNX-IR conversion pipeline
///
/// These tests verify the correctness of the ONNX-IR infrastructure:
/// - Value source tracking (Static/Constant/Dynamic)
/// - Type system (Scalar/Shape/Tensor, multiple DTypes)
/// - Graph structure handling (branching, multiple I/O)
/// - Pipeline phases (Identity elimination, constant removal)
///
/// These tests focus on HOW the pipeline works, not WHAT operations it supports.
mod test_utils;

use test_utils::*;

// ============================================================================
// Constant Handling Tests
// ============================================================================

#[test]
fn test_constant_value_sources() {
    // Test Static vs Constant vs Dynamic value source tracking
    let graph = load_onnx("constants.onnx");

    // Should have 1 runtime input (Dynamic)
    assert_eq!(graph.inputs.len(), 1, "Expected 1 runtime input");

    // Should have operation nodes (Add, Mul)
    assert_eq!(
        count_operation_nodes(&graph),
        2,
        "Expected 2 operation nodes"
    );

    // Unused constant should be removed in Phase 5
    assert!(
        count_constant_nodes(&graph) <= 2,
        "Unused constants should be removed"
    );
}

#[test]
fn test_unreferenced_constant_removal() {
    // The constants model has 3 initializers but only 2 are used
    // Phase 5 should remove the unreferenced one
    let graph = load_onnx("constants.onnx");

    let const_count = count_constant_nodes(&graph);
    println!(
        "Constant nodes after finalization: {} (max 2 expected)",
        const_count
    );

    assert!(const_count <= 2, "Unreferenced constant should be removed");
}

// ============================================================================
// Type System Tests
// ============================================================================

#[test]
fn test_multiple_data_types() {
    // Test that different data types are preserved through the pipeline
    let graph = load_onnx("data_types.onnx");

    assert_eq!(graph.inputs.len(), 4, "Expected 4 inputs");
    assert_eq!(graph.outputs.len(), 5, "Expected 5 outputs");

    // Verify type diversity
    let input_types = get_input_dtypes(&graph);
    let unique_types: std::collections::HashSet<_> = input_types.into_iter().collect();

    println!("Unique data types: {}", unique_types.len());
    assert!(
        unique_types.len() >= 3,
        "Should have multiple data types (F32, F64, I32, I64)"
    );
}

#[test]
fn test_argument_types() {
    // Test Scalar, Shape, and Tensor argument types
    let graph = load_onnx("arg_types.onnx");

    assert_eq!(graph.inputs.len(), 3, "Expected 3 inputs");
    assert_eq!(graph.outputs.len(), 3, "Expected 3 outputs");

    let (scalars, shapes, tensors) = count_outputs_by_type(&graph);

    println!(
        "Output types - Scalars: {}, Shapes: {}, Tensors: {}",
        scalars, shapes, tensors
    );

    // Should have at least one of each type
    assert_eq!(scalars, 1, "Should have 1 scalar output");
    assert_eq!(shapes, 1, "Should have 1 shape output");
    assert_eq!(tensors, 1, "Should have 1 tensor output");
}

// ============================================================================
// Graph Structure Tests
// ============================================================================

#[test]
fn test_graph_branching() {
    // Test that one output consumed by multiple nodes works correctly
    let graph = load_onnx("branching.onnx");

    assert_eq!(graph.inputs.len(), 1, "Expected 1 input");
    assert_eq!(graph.outputs.len(), 3, "Expected 3 outputs from branching");

    // Should have Relu as the branch point
    assert!(
        has_node_type(&graph, onnx_ir::ir::NodeType::Relu),
        "Should have Relu node as branch point"
    );

    // Should have at least 4 nodes (Relu + 3 consumers)
    assert!(graph.nodes.len() >= 4, "Expected branching structure");
}

#[test]
fn test_multiple_inputs_outputs() {
    // Test proper handling of multiple graph inputs and outputs
    let graph = load_onnx("multi_io.onnx");

    assert_eq!(graph.inputs.len(), 3, "Expected 3 inputs");
    assert_eq!(graph.outputs.len(), 2, "Expected 2 outputs");

    // All inputs should be accessible
    for (i, input) in graph.inputs.iter().enumerate() {
        assert!(!input.name.is_empty(), "Input {} should have a name", i);
    }

    // All outputs should reference valid nodes
    for (i, output) in graph.outputs.iter().enumerate() {
        assert!(!output.name.is_empty(), "Output {} should have a name", i);
    }
}

// ============================================================================
// Pipeline Phase Tests
// ============================================================================

#[test]
fn test_identity_elimination() {
    // Test Phase 4: Identity nodes should be eliminated
    let graph = load_onnx("identity.onnx");

    let identity_count = count_nodes(&graph, onnx_ir::ir::NodeType::Identity);

    println!(
        "Identity nodes after Phase 4: {} (should be 0)",
        identity_count
    );

    assert_eq!(
        identity_count, 0,
        "All Identity nodes should be eliminated in Phase 4"
    );

    // Should have exactly 2 operation nodes (Relu and Add)
    assert_eq!(
        count_operation_nodes(&graph),
        2,
        "Should have 2 operation nodes after Identity elimination"
    );
}

#[test]
fn test_transitive_rewiring() {
    // Test that chains of Identity nodes are properly rewired
    let graph = load_onnx("identity.onnx");

    // The original ONNX has: relu → identity1 → identity2 → add → identity3 → output
    // After rewiring: relu → add → output

    assert!(
        has_node_type(&graph, onnx_ir::ir::NodeType::Relu),
        "Relu should remain"
    );
    assert!(
        has_node_type(&graph, onnx_ir::ir::NodeType::Add),
        "Add should remain"
    );
    assert!(
        !has_node_type(&graph, onnx_ir::ir::NodeType::Identity),
        "Identity should be eliminated"
    );

    // Output should be properly rewired to Add's output
    assert_eq!(graph.outputs.len(), 1, "Should have 1 output");
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

#[test]
fn test_scalar_representations() {
    // Test different scalar representations (empty shape [], shape [1], shape [1,1])
    let graph = load_onnx("edge_cases.onnx");

    // Should successfully parse without panicking
    assert_eq!(graph.inputs.len(), 1);
    assert_eq!(graph.outputs.len(), 3);

    // Should have 3 operation nodes using the scalar constants
    assert_eq!(
        count_operation_nodes(&graph),
        3,
        "Should have 3 operations with scalar constants"
    );
}
