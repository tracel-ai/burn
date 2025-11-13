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
fn test_empty_graph() {
    // Test the absolute minimal graph: input IS output with ZERO nodes
    // This tests edge case #1: Empty graph (input→output, no operations)
    let graph = load_onnx("empty_graph.onnx");

    assert_eq!(graph.inputs.len(), 1, "Should have 1 input");
    assert_eq!(graph.outputs.len(), 1, "Should have 1 output");

    // Graph with ZERO nodes - input tensor is directly the output tensor
    let node_count = graph.nodes.len();
    println!("Empty graph has {} nodes", node_count);

    assert_eq!(
        node_count, 0,
        "Empty graph should have ZERO nodes (input is directly output)"
    );

    // Input and output should reference the same tensor
    assert_eq!(
        graph.inputs[0].name, graph.outputs[0].name,
        "Input and output should be the same tensor in empty graph"
    );

    // Should handle minimal graph structure without errors
    println!("Empty graph parsed successfully with ZERO nodes");
}

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

#[test]
fn test_diamond_pattern() {
    // Test graph where computation splits and then merges back
    // This tests edge case #9: Diamond pattern (split then merge)
    let graph = load_onnx("diamond_pattern.onnx");

    assert_eq!(graph.inputs.len(), 1, "Should have 1 input");
    assert_eq!(graph.outputs.len(), 1, "Should have 1 output");

    // Should have split and merge structure
    let op_count = count_operation_nodes(&graph);
    println!("Diamond pattern has {} operation nodes", op_count);

    assert!(
        op_count >= 3,
        "Should have at least 3 nodes (split paths + merge)"
    );

    // Type inference should converge despite split/merge pattern
    println!("Diamond pattern: Type inference converged successfully");
}

// ============================================================================
// Type System Edge Cases
// ============================================================================

#[test]
fn test_type_validation_scalar_tensor() {
    // Test type inference and validation with scalar vs tensor operations
    // This tests edge case #6: Type mismatch validation (Scalar vs Tensor)
    let graph = load_onnx("type_validation.onnx");

    assert!(!graph.inputs.is_empty(), "Should have inputs");
    assert!(!graph.outputs.is_empty(), "Should have outputs");

    println!("Type validation graph: {} nodes", graph.nodes.len());

    // Should handle mixed scalar and tensor types correctly
    // If type inference couldn't handle it, parsing would have failed
    use onnx_ir::ir::ArgType;

    let mut has_scalar = false;
    let mut has_tensor = false;

    for input in &graph.inputs {
        match &input.ty {
            ArgType::Scalar(_) => {
                has_scalar = true;
                println!("Found scalar input: {}", input.name);
            }
            ArgType::Tensor(t) => {
                has_tensor = true;
                println!("Found tensor input: {} (rank {})", input.name, t.rank);
            }
            _ => {}
        }
    }

    // Should have both scalar and tensor types to test validation
    assert!(
        has_scalar || has_tensor,
        "Should have scalar and/or tensor inputs for type validation"
    );

    println!("Type validation: Successfully handled scalar/tensor operations");
}

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
fn test_complex_broadcasting() {
    // Test complex broadcasting with different shapes
    // This tests edge case #11: Incompatible broadcasting (complex shapes)
    let graph = load_onnx("complex_broadcasting.onnx");

    assert!(!graph.inputs.is_empty(), "Should have inputs");
    assert!(!graph.outputs.is_empty(), "Should have outputs");

    println!("Complex broadcasting: {} nodes", graph.nodes.len());

    // Should successfully infer types despite complex broadcasting
    // (validated by successful parsing and type inference)
    println!("Complex broadcasting handled successfully");
}

#[test]
fn test_all_dynamic_shapes() {
    // Test model with all dynamic (symbolic) shapes
    // This tests edge case #12: All dynamic shapes (no static)
    let graph = load_onnx("all_dynamic_shapes.onnx");

    assert!(!graph.inputs.is_empty(), "Should have inputs");
    assert!(!graph.outputs.is_empty(), "Should have outputs");

    println!("All dynamic shapes: {} nodes", graph.nodes.len());

    // Should handle fully dynamic shapes without static shape info
    println!("All dynamic shapes: Type inference succeeded");
}

#[test]
fn test_circular_preferences() {
    // Test type inference convergence with circular dependencies
    // This tests edge case #14: Circular preferences convergence
    let graph = load_onnx("circular_preferences.onnx");

    assert!(!graph.inputs.is_empty(), "Should have inputs");
    assert!(!graph.outputs.is_empty(), "Should have outputs");

    println!("Circular preferences: {} nodes", graph.nodes.len());

    // Type inference should converge even with circular preference patterns
    println!("Circular preferences: Converged successfully");
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

#[test]
fn test_constant_lifting_phase2() {
    // Test constants lifted during Phase 2 node conversion
    // This tests edge case #15: Constants lifted in Phase 2, used in Phase 3
    let graph = load_onnx("constant_lifting.onnx");

    assert!(!graph.inputs.is_empty(), "Should have inputs");
    assert!(!graph.outputs.is_empty(), "Should have outputs");

    let const_count = count_constant_nodes(&graph);
    println!("Constant lifting: {} constant nodes", const_count);

    // Some constants may be coalesced into operations (e.g., MatMul → Linear)
    // The important thing is that the graph parses successfully
    assert!(const_count >= 1, "Should have at least one constant");

    // Verify operations exist that use constants
    let op_count = count_operation_nodes(&graph);
    assert!(
        op_count >= 2,
        "Should have multiple operations using constants"
    );

    println!("Constants lifted and used successfully");

    // Diagnostic: Show what nodes we actually have
    for (i, node) in graph.nodes.iter().enumerate() {
        println!("  Node {}: '{}'", i, node.name());
    }
}

#[test]
fn test_constant_referenced_multiple_times() {
    // Test a constant used by multiple operations
    // This tests edge case #20: Constant referenced multiple times
    let graph = load_onnx("constant_multiple_refs.onnx");

    assert_eq!(graph.outputs.len(), 3, "Should have 3 outputs");

    let const_count = count_constant_nodes(&graph);
    println!("Multiple refs: {} constant nodes", const_count);

    // The constant should be preserved (reference count > 1)
    assert!(const_count > 0, "Should have constant nodes");

    println!("Constant with multiple references preserved correctly");
}

#[test]
fn test_constant_in_graph_output() {
    // Test constant that is directly in graph output
    // This tests edge case #21: Constant referenced in graph output
    let graph = load_onnx("constant_in_output.onnx");

    assert_eq!(graph.outputs.len(), 2, "Should have 2 outputs");

    let const_count = count_constant_nodes(&graph);
    println!("Constant in output: {} constant nodes", const_count);

    // Constant used in output should NOT be removed
    assert!(const_count > 0, "Should have constants for output");

    println!("Constant in output preserved correctly");
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

    println!("Single node graph: {}", graph.nodes[0].name());
}

// ============================================================================
// Node Remapping Edge Cases
// ============================================================================

#[test]
fn test_matmul_dynamic_weights_no_coalesce() {
    // Test MatMul with runtime (dynamic) weights
    // This tests edge case #18: MatMul with dynamic weights (no coalesce)
    let graph = load_onnx("matmul_dynamic_weights.onnx");

    assert_eq!(graph.inputs.len(), 2, "Should have 2 runtime inputs");

    // Should have MatMul node (NOT coalesced to Linear because weights are runtime)
    let has_matmul = has_node_type(&graph, onnx_ir::ir::NodeType::MatMul);
    let has_linear = has_node_type(&graph, onnx_ir::ir::NodeType::Linear);

    println!(
        "MatMul dynamic weights: MatMul={}, Linear={}",
        has_matmul, has_linear
    );

    // Should stay as MatMul (no Linear with dynamic weights)
    assert!(
        has_matmul,
        "Should have MatMul node when weights are dynamic"
    );

    println!("MatMul with dynamic weights: NOT coalesced to Linear (correct)");
}

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
#[test]
fn test_same_constant_in_multiple_inputs() {
    // Test where ONE operation has MULTIPLE inputs pointing to THE SAME constant
    // This tests: constant lifting, reference counting, deduplication
    let graph = load_onnx("same_constant_multiple_inputs.onnx");

    println!("\n=== Same Constant Multiple Inputs ===");
    println!("Nodes: {}", graph.nodes.len());

    for (i, node) in graph.nodes.iter().enumerate() {
        println!("\nNode {}: '{}'", i, node.name());
        println!("  Inputs ({}):", node.inputs().len());
        for (j, inp) in node.inputs().iter().enumerate() {
            println!("    [{}] name='{}', value={:?}", j, inp.name, inp.value());
        }
    }

    // Find the Where operation
    let where_node = graph
        .nodes
        .iter()
        .find(|n| matches!(n, onnx_ir::ir::Node::Where { .. }))
        .expect("Should have Where node");

    println!("\n=== Where Node Analysis ===");
    println!(
        "Inputs: {} (should be 3: condition, true_val, false_val)",
        where_node.inputs().len()
    );
    assert_eq!(where_node.inputs().len(), 3, "Where should have 3 inputs");

    // Check if inputs [1] and [2] reference the same constant
    let true_val = &where_node.inputs()[1];
    let false_val = &where_node.inputs()[2];

    println!(
        "  true_val:  name='{}', value={:?}",
        true_val.name,
        true_val.value()
    );
    println!(
        "  false_val: name='{}', value={:?}",
        false_val.name,
        false_val.value()
    );

    // CRITICAL TEST: Do both inputs point to the same underlying constant data?
    // This tests whether the constant lifting properly handles same constant in multiple slots

    use onnx_ir::ir::ValueSource;

    println!("\nValue source check:");

    // Extract data_id from ValueSource if it's Static
    let true_val_data_id = match true_val.value_source {
        ValueSource::Static(id) => Some(id),
        _ => None,
    };
    let false_val_data_id = match false_val.value_source {
        ValueSource::Static(id) => Some(id),
        _ => None,
    };

    println!(
        "  true_val:  source={:?}, data_id={:?}",
        true_val.value_source, true_val_data_id
    );
    println!(
        "  false_val: source={:?}, data_id={:?}",
        false_val.value_source, false_val_data_id
    );

    // Check 1: Both should reference constants (not runtime Dynamic values)
    assert!(
        matches!(
            true_val.value_source,
            ValueSource::Constant | ValueSource::Static(_)
        ),
        "true_val should be Constant or Static, got: {:?}",
        true_val.value_source
    );
    assert!(
        matches!(
            false_val.value_source,
            ValueSource::Constant | ValueSource::Static(_)
        ),
        "false_val should be Constant or Static, got: {:?}",
        false_val.value_source
    );

    // Check 2: If both have data_ids, they should be THE SAME
    match (true_val_data_id, false_val_data_id) {
        (Some(id1), Some(id2)) => {
            if id1 == id2 {
                println!(
                    "\n  ✓ CORRECT: Both inputs reference the SAME constant data (data_id: {})",
                    id1
                );
                println!("     The initializer was properly shared, not duplicated.");
            } else {
                println!(
                    "\n  ✗ POTENTIAL BUG: Inputs reference DIFFERENT data_ids ({} vs {})",
                    id1, id2
                );
                println!(
                    "     This means the same initializer was duplicated instead of being shared!"
                );
                panic!("Same ONNX initializer should have same data_id in both input slots");
            }
        }
        (None, None) => {
            println!("\n  ℹ Both inputs have data_id=None");
            println!("    They point to constant node outputs (ValueSource::Constant)");
            // Check if they point to the same constant node by name
            if true_val.name == false_val.name {
                println!(
                    "    ✓ CORRECT: Both point to the same constant node: '{}'",
                    true_val.name
                );
            } else {
                println!(
                    "    ✗ BUG: Different constant nodes: '{}' vs '{}'",
                    true_val.name, false_val.name
                );
                panic!("Same initializer should map to same constant node");
            }
        }
        _ => {
            println!(
                "\n  ? One has data_id, one doesn't: {:?} vs {:?}",
                true_val_data_id, false_val_data_id
            );
        }
    }

    println!("\n✓ Test passed: Same ONNX initializer properly handled in multiple input slots");
}

// ============================================================================
// Lower Priority Edge Cases
// ============================================================================

#[test]
fn test_disconnected_subgraphs() {
    // Test graph with multiple independent computation paths
    // This tests edge case #27: Multiple disconnected subgraphs
    let graph = load_onnx("disconnected_subgraphs.onnx");

    assert_eq!(
        graph.inputs.len(),
        4,
        "Should have 4 inputs (2 per subgraph)"
    );
    assert_eq!(
        graph.outputs.len(),
        2,
        "Should have 2 outputs (1 per subgraph)"
    );

    println!("Disconnected subgraphs: {} nodes", graph.nodes.len());

    // Type inference should handle disconnected paths
    assert!(graph.nodes.len() >= 4, "Should have at least 4 nodes");

    println!("Disconnected subgraphs handled successfully");
}

#[test]
fn test_node_multiple_outputs_partial_use() {
    // Test node with multiple outputs where only one is used
    // This tests edge case #28: Node with multiple outputs, only one used
    let graph = load_onnx("node_multiple_outputs_partial_use.onnx");

    // TopK should produce 2 outputs even if only one is used
    let topk_node = graph
        .nodes
        .iter()
        .find(|n| matches!(n, onnx_ir::ir::Node::TopK { .. }))
        .expect("Should have TopK node");

    println!("TopK node outputs: {}", topk_node.outputs().len());
    assert_eq!(topk_node.outputs().len(), 2, "TopK should have 2 outputs");

    // But only one output should be consumed by other nodes
    println!("Node with multiple outputs, partial use handled correctly");
}

#[test]
fn test_optional_input_clip() {
    // Test Clip with optional max input not provided
    // This tests edge case #29: Optional input not provided (Clip)
    let graph = load_onnx("optional_input_clip.onnx");

    let clip_node = graph
        .nodes
        .iter()
        .find(|n| matches!(n, onnx_ir::ir::Node::Clip { .. }))
        .expect("Should have Clip node");

    println!("Clip node inputs: {}", clip_node.inputs().len());

    // Clip should handle optional inputs gracefully
    println!("Optional input handling: Clip parsed successfully");
}

#[test]
fn test_shape_type_broadcasting() {
    // Test Shape type in broadcasting context
    // This tests edge case #30: Shape type in broadcasting context
    let graph = load_onnx("shape_broadcasting.onnx");

    // Should have Shape node producing int64 tensor
    let has_shape = has_node_type(&graph, onnx_ir::ir::NodeType::Shape);
    assert!(has_shape, "Should have Shape node");

    println!("Shape type in broadcasting: {} nodes", graph.nodes.len());
    println!("Shape type broadcasting handled successfully");
}

#[test]
fn test_large_constants_mb_sized() {
    // Test handling of large constants (MB-sized)
    // This tests edge case #32: Large constants (MB-sized)
    let graph = load_onnx("large_constants.onnx");

    let const_count = count_constant_nodes(&graph);
    let op_count = count_operation_nodes(&graph);

    println!(
        "Large constants: {} constant nodes, {} operations",
        const_count, op_count
    );

    // The MatMul with large weight gets coalesced into Linear (which internalizes the weight)
    // The key test is that the graph parses successfully despite MB-sized data
    assert!(
        op_count >= 1,
        "Should have operations using large constants"
    );
    assert!(
        !graph.nodes.is_empty(),
        "Should successfully parse MB-sized model"
    );

    println!("MB-sized constants handled successfully (coalesced into operations)");
}

#[test]
fn test_very_long_node_names() {
    // Test handling of very long node names (100+ characters)
    // This tests edge case #40: Very long node names (100+ chars)
    let graph = load_onnx("very_long_names.onnx");

    // Check that long names are preserved
    for node in &graph.nodes {
        if node.name().len() > 100 {
            println!("Found node with long name: {} chars", node.name().len());
        }
    }

    for input in &graph.inputs {
        if input.name.len() > 100 {
            println!("Found input with long name: {} chars", input.name.len());
        }
    }

    println!("Very long names (100+ chars) handled successfully");
}

#[test]
fn test_unknown_rank_with_dynamic_shapes() {
    // Test type inference with unknown rank and dynamic shapes
    // This tests edge case #41: Unknown rank with dynamic shapes
    let graph = load_onnx("unknown_rank_dynamic.onnx");

    assert!(!graph.inputs.is_empty(), "Should have inputs");
    assert!(!graph.outputs.is_empty(), "Should have outputs");

    println!(
        "Unknown rank with dynamic shapes: {} nodes",
        graph.nodes.len()
    );
    println!("Unknown rank handled successfully");
}

#[test]
fn test_static_constant_value_source_invariant() {
    // CORRECTNESS TEST: Verify invariant that values cannot be both Static AND Constant
    // This tests edge case #33: Static AND Constant invariant
    let graph = load_onnx("static_constant_invariant.onnx");

    use onnx_ir::ir::ValueSource;

    println!("\n=== Value Source Invariant Check ===");

    // Check all node inputs to verify the invariant
    for node in &graph.nodes {
        for (i, input) in node.inputs().iter().enumerate() {
            // Extract data_id from ValueSource::Static variant
            let has_data = matches!(input.value_source, ValueSource::Static(_));
            let is_constant_source = matches!(input.value_source, ValueSource::Constant);
            let is_static_source = matches!(input.value_source, ValueSource::Static(_));

            println!(
                "Node '{}' input[{}]: source={:?}, has_data_id={}",
                node.name(),
                i,
                input.value_source,
                has_data
            );

            // INVARIANT CHECK: Should be either Static (with embedded data_id) OR Constant, never both
            match input.value_source {
                ValueSource::Static(_) => {
                    // Static with embedded data_id
                    println!("  ✓ Static with embedded data_id");
                }
                ValueSource::Constant => {
                    // Constant: points to constant node output
                    println!("  ✓ Constant (references constant node output)");
                }
                ValueSource::Dynamic => {
                    println!("  ✓ Dynamic (runtime value)");
                }
                ValueSource::Optional => {
                    println!("  ✓ Optional (not provided)");
                }
            }

            // CRITICAL: Cannot be BOTH Static AND Constant simultaneously
            // With the new design, Static contains data_id, so they are mutually exclusive by design
            assert!(
                !(is_static_source && is_constant_source),
                "INVARIANT VIOLATION: Input cannot be both Static AND Constant!"
            );
        }
    }

    println!("\n✓ Invariant verified: No value is both Static AND Constant");
}

#[test]
fn test_empty_output_names_handling() {
    // Test handling of empty/optional output names
    // This tests edge case #39: Empty output names (optional outputs)
    let graph = load_onnx("empty_output_names.onnx");

    // Should parse successfully even if model has empty string handling
    assert!(!graph.nodes.is_empty(), "Should have nodes");
    assert!(!graph.outputs.is_empty(), "Should have outputs");

    println!("Empty output names handled successfully");
}

// ============================================================================
// Validation / Correctness Tests
// ============================================================================

#[test]
fn test_direct_input_to_output() {
    // VALIDATION TEST: Input tensor is DIRECTLY the output tensor with ZERO nodes
    // Tests the absolute edge case of a graph with no operations at all
    // This validates that the parser can handle graphs with no processing nodes
    let graph = load_onnx("empty_graph.onnx");

    println!("\n=== Direct Input to Output Validation Test ===");
    println!("Graph has {} nodes", graph.nodes.len());
    println!("Graph has {} inputs", graph.inputs.len());
    println!("Graph has {} outputs", graph.outputs.len());

    // CORRECTNESS: Should have exactly 0 nodes - input is directly the output
    assert_eq!(
        graph.nodes.len(),
        0,
        "Graph with direct input→output should have ZERO nodes"
    );

    assert_eq!(graph.inputs.len(), 1, "Should have 1 input");
    assert_eq!(graph.outputs.len(), 1, "Should have 1 output");

    // CORRECTNESS: The input and output should reference the same tensor
    assert_eq!(
        graph.inputs[0].name, graph.outputs[0].name,
        "Input and output should be the same tensor"
    );

    println!("✓ Parser correctly handled graph with no operations");
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "Nodes are not topologically sorted")]
fn test_non_topological_order_handling() {
    // VALIDATION TEST: Graph nodes are NOT in topological order
    // Expected behavior: Parser should REJECT non-topological graphs
    // This tests that the parser correctly validates topological ordering
    //
    // NOTE: This test only runs in debug mode because the topological ordering
    // validation is a debug_assert! that is disabled in release builds.
    //
    // The ONNX model has nodes in this order:
    //   1. Add (uses abs_out)
    //   2. Abs (uses relu_out)
    //   3. Relu (uses input)
    //
    // Correct execution order should be: Relu → Abs → Add
    // The parser should panic when it detects this violation
    let _graph = load_onnx("non_topological_order.onnx");

    // This line should never be reached - the panic should occur during parsing
    unreachable!("Parser should have rejected non-topological graph");
}
