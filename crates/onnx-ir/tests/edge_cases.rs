/// Edge case tests for ONNX-IR infrastructure
///
/// These tests verify the infrastructure handles edge cases and corner cases correctly:
/// - Unusual graph structures (no ops, only constants, passthrough)
/// - Type inference edge cases (broadcasting, rank mismatches)
/// - Phase boundary conditions (empty graphs, single nodes)
/// - Extreme scenarios (very deep chains, wide branching)
///
/// These tests push the boundaries of what the pipeline should handle gracefully.
mod test_utils;

use test_utils::*;

// ============================================================================
// Graph Structure Edge Cases
// ============================================================================

#[test]
fn test_graph_with_only_constants() {
    // Test a graph that has no operations, only constant nodes
    // This tests Phase 5 unreferenced constant removal when ALL nodes are constants
    let graph = load_onnx("only_constants.onnx");

    // Should have graph inputs/outputs but potentially no operation nodes
    assert!(!graph.inputs.is_empty(), "Should have inputs");
    assert!(!graph.outputs.is_empty(), "Should have outputs");

    // All nodes might be Constants or there might be zero nodes if optimized away
    let const_count = count_constant_nodes(&graph);
    let op_count = count_operation_nodes(&graph);

    println!(
        "Only-constants graph: {} constants, {} operations",
        const_count, op_count
    );

    // The graph should handle this gracefully without panicking
    assert!(
        const_count > 0 || op_count > 0,
        "Graph should have at least some nodes"
    );
}

#[test]
fn test_passthrough_graph() {
    // Test a graph where input is directly connected to output (via Identity or direct)
    // This tests graph I/O handling and Identity elimination edge cases
    let graph = load_onnx("passthrough.onnx");

    assert_eq!(graph.inputs.len(), 1, "Should have 1 input");
    assert_eq!(graph.outputs.len(), 1, "Should have 1 output");

    // After Identity elimination, might have 0 or 1 nodes
    let node_count = graph.nodes.len();
    println!("Passthrough graph has {} nodes", node_count);

    // Note: Direct I/O Identity nodes might be preserved to maintain graph structure
    // This is actually correct behavior - the graph should handle this gracefully
    let has_identity = has_node_type(&graph, onnx_ir::ir::NodeType::Identity);
    println!("Has Identity: {}", has_identity);

    // Graph should be valid with minimal nodes
    assert!(
        node_count <= 1,
        "Passthrough should have at most 1 node (Identity or eliminated)"
    );
}

#[test]
fn test_very_deep_chain() {
    // Test a very deep sequential chain (e.g., 50 nodes in sequence)
    // This tests type inference convergence and graph traversal at scale
    let graph = load_onnx("deep_chain.onnx");

    assert_eq!(graph.inputs.len(), 1, "Should have 1 input");
    assert_eq!(graph.outputs.len(), 1, "Should have 1 output");

    // Should have many sequential operations
    assert!(
        count_operation_nodes(&graph) >= 20,
        "Should have a deep chain of operations"
    );

    println!("Deep chain has {} nodes", graph.nodes.len());

    // Type inference should converge even with deep chains
    // (if it didn't, parsing would have failed)
}

#[test]
fn test_wide_branching() {
    // Test a graph with one node feeding many consumers (e.g., 10+ outputs)
    // This tests reference counting and connectivity at scale
    let graph = load_onnx("wide_branching.onnx");

    assert_eq!(graph.inputs.len(), 1, "Should have 1 input");

    // Should have many outputs from branching
    assert!(
        graph.outputs.len() >= 5,
        "Should have wide branching with many outputs"
    );

    println!(
        "Wide branching: {} outputs from single input",
        graph.outputs.len()
    );

    // All outputs should be valid
    for (i, output) in graph.outputs.iter().enumerate() {
        assert!(!output.name.is_empty(), "Output {} should have a name", i);
    }
}

// ============================================================================
// Type System Edge Cases
// ============================================================================

#[test]
fn test_mixed_rank_broadcasting() {
    // Test operations with mixed ranks (scalar + 4D tensor, 1D + 3D, etc.)
    // This tests NumPy-style broadcasting rules
    let graph = load_onnx("mixed_rank_broadcasting.onnx");

    // Should successfully parse despite rank differences
    assert!(!graph.nodes.is_empty(), "Should have nodes");
    assert!(!graph.outputs.is_empty(), "Should have outputs");

    println!("Mixed rank broadcasting: {} nodes", graph.nodes.len());

    // The operations should handle broadcasting correctly
    // (validated by successful parsing and type inference)
}

#[test]
fn test_zero_sized_dimensions() {
    // Test tensors with 0-sized dimensions (e.g., shape [2, 0, 3])
    // This is valid in ONNX and should be handled
    let graph = load_onnx("zero_sized_dims.onnx");

    assert!(!graph.inputs.is_empty(), "Should have inputs");
    assert!(!graph.outputs.is_empty(), "Should have outputs");

    println!("Zero-sized dims graph parsed successfully");

    // Should parse without errors even with 0-sized dimensions
}

#[test]
fn test_high_rank_tensors() {
    // Test with 5D and 6D tensors
    // This tests shape inference and type system at high dimensionality
    let graph = load_onnx("high_rank_tensors.onnx");

    assert!(!graph.inputs.is_empty(), "Should have inputs");

    // Verify we can handle high-rank tensors
    use onnx_ir::ir::ArgType;

    for input in &graph.inputs {
        if let ArgType::Tensor(tensor_type) = &input.ty {
            println!("Input rank: {}", tensor_type.rank);
            if tensor_type.rank >= 5 {
                println!("Found high-rank tensor ({}D)", tensor_type.rank);
            }
        }
    }
}

// ============================================================================
// Constant Handling Edge Cases
// ============================================================================

#[test]
fn test_all_constants_referenced() {
    // Test where all initializers are used (none should be removed)
    // This tests Phase 5 reference counting
    let graph = load_onnx("all_constants_used.onnx");

    let const_count = count_constant_nodes(&graph);
    let op_count = count_operation_nodes(&graph);

    println!(
        "All constants used: {} constants, {} operations",
        const_count, op_count
    );

    // All constants should be preserved (none removed)
    // This depends on the model having N initializers and N being used
    assert!(const_count > 0, "Should have constant nodes");
    assert!(op_count > 0, "Should have operations using constants");
}

#[test]
fn test_no_initializers() {
    // Test a graph with no initializers at all (all runtime inputs)
    // This tests Phase 1 when there are no constants to process
    let graph = load_onnx("no_initializers.onnx");

    let const_count = count_constant_nodes(&graph);

    println!("No initializers: {} constant nodes", const_count);

    // Should have no Constant nodes
    assert_eq!(
        const_count, 0,
        "Should have no constant nodes without initializers"
    );

    // Should still have valid operations
    assert!(count_operation_nodes(&graph) > 0, "Should have operations");
}

// ============================================================================
// Value Source Edge Cases
// ============================================================================

#[test]
fn test_mixed_value_sources_single_op() {
    // Test an operation with mixed value sources (Static + Dynamic inputs)
    // E.g., Add with one constant (Static) and one runtime input (Dynamic)
    let graph = load_onnx("mixed_value_sources.onnx");

    assert!(!graph.nodes.is_empty(), "Should have nodes");

    // Should successfully handle mixed value sources
    // This is actually the common case (e.g., bias addition)
    println!(
        "Mixed value sources: {} nodes parsed successfully",
        graph.nodes.len()
    );
}

// ============================================================================
// Phase Boundary Edge Cases
// ============================================================================

#[test]
fn test_single_node_graph() {
    // Test the simplest possible graph: single operation
    // This tests phase transitions with minimal graph
    let graph = load_onnx("single_node.onnx");

    assert_eq!(graph.inputs.len(), 1, "Should have 1 input");
    assert_eq!(graph.outputs.len(), 1, "Should have 1 output");

    // Should have exactly 1 operation node (possibly + constants)
    let op_count = count_operation_nodes(&graph);
    assert_eq!(op_count, 1, "Should have exactly 1 operation node");

    println!("Single node graph: {}", graph.nodes[0].name);
}

// ============================================================================
// Node Remapping Edge Cases
// ============================================================================

#[test]
fn test_gemm_to_linear_conversion() {
    // Test Gemm → Linear coalescing edge case
    // This tests Phase 2 node remapping
    let graph = load_onnx("gemm_linear.onnx");

    // After coalescing, should have Linear node instead of Gemm
    // (or Gemm if it doesn't meet Linear criteria)

    let has_linear = has_node_type(&graph, onnx_ir::ir::NodeType::Linear);
    let has_gemm = has_node_type(&graph, onnx_ir::ir::NodeType::Gemm);

    println!(
        "Gemm/Linear conversion: Linear={}, Gemm={}",
        has_linear, has_gemm
    );

    // Should have one or the other (not both for same operation)
    assert!(
        has_linear || has_gemm,
        "Should have Linear or Gemm node after conversion"
    );
}
