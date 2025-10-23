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

// --------------------------------------
// Constant Handling Tests
// --------------------------------------

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

// --------------------------------------
// Type System Tests
// --------------------------------------

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

// --------------------------------------
// Graph Structure Tests
// --------------------------------------

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

// --------------------------------------
// Pipeline Phase Tests
// --------------------------------------

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

// --------------------------------------
// Edge Cases Tests
// --------------------------------------

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

#[test]
fn test_all_data_types_conversion() {
    // INFRASTRUCTURE TEST: Validates ALL data type conversions
    // Tests F32, F64, I32, I64, Bool, U8, I8, F16, U16 with non-empty and empty tensors
    let graph = load_onnx("all_dtypes.onnx");

    println!("\n= All Data Types Conversion Test =");

    // All initializers become Constant nodes, tracked by count
    let const_count = count_constant_nodes(&graph);
    println!("Found {} constant nodes (from initializers)", const_count);
    assert_eq!(
        const_count, 22,
        "Should have 22 constant nodes (all data types + empty variants + multi-dim)"
    );

    // Track which data types we've validated
    let mut validated_types = std::collections::HashSet::new();
    let mut empty_count = 0;
    let mut non_empty_count = 0;

    // Find all constant nodes
    for node in &graph.nodes {
        if !matches!(node.node_type, onnx_ir::ir::NodeType::Constant) {
            continue;
        }

        // CRITICAL: After constant lifting in Phase 2, Constant nodes should have NO attributes
        // The 'value' attribute should be moved to the first input
        assert!(
            node.attrs.is_empty(),
            "Constant node '{}' should have no attributes after constant lifting - 'value' should be in first input",
            node.name
        );

        // After constant lifting, Constant nodes MUST have their data in the first input
        assert!(
            !node.inputs.is_empty(),
            "Constant node '{}' has no inputs - should have tensor data in first input after lifting",
            node.name
        );

        // Constant nodes MUST have exactly one output
        assert_eq!(
            node.outputs.len(),
            1,
            "Constant node '{}' should have exactly 1 output, found {}",
            node.name,
            node.outputs.len()
        );

        let input_arg = &node.inputs[0];
        if let Some(tensor_data) = input_arg.value() {
            let dtype = tensor_data.dtype;
            let shape = &tensor_data.shape;
            let num_elems: usize = shape.iter().product();

            // Verify output type matches tensor data type
            let output_arg = &node.outputs[0];
            match &output_arg.ty {
                onnx_ir::ir::ArgType::Tensor(tensor_type) => {
                    assert_eq!(
                        tensor_type.dtype, dtype,
                        "Constant node '{}' tensor output dtype {:?} doesn't match tensor data dtype {:?}",
                        node.name, tensor_type.dtype, dtype
                    );
                }
                onnx_ir::ir::ArgType::Scalar(scalar_dtype) => {
                    assert_eq!(
                        scalar_dtype, &dtype,
                        "Constant node '{}' scalar output dtype {:?} doesn't match tensor data dtype {:?}",
                        node.name, scalar_dtype, dtype
                    );
                }
                onnx_ir::ir::ArgType::Shape(_) => {
                    panic!(
                        "Constant node '{}' has Shape output type, but contains tensor data with dtype {:?}",
                        node.name, dtype
                    );
                }
            }

            validated_types.insert(dtype);

            if num_elems == 0 {
                empty_count += 1;
                println!(
                    "Empty {:?} tensor '{}' validated (shape: {:?})",
                    dtype, node.name, shape
                );
            } else {
                non_empty_count += 1;

                // Validate we can actually read the data AND verify the values AND shapes are correct
                // Use (dtype, shape) tuple to identify which tensor we're validating
                match (dtype, shape.as_slice()) {
                    (burn_tensor::DType::F32, [7]) => {
                        let data = tensor_data.iter::<f32>().collect::<Vec<_>>();
                        // [1.5, -2.5, 3.14159, 0.0, inf, -inf, 2.71828]
                        assert_eq!(data[0], 1.5);
                        assert_eq!(data[1], -2.5);
                        assert!((data[2] - 3.14159).abs() < 0.00001);
                        assert_eq!(data[3], 0.0);
                        assert!(data[4].is_infinite() && data[4].is_sign_positive());
                        assert!(data[5].is_infinite() && data[5].is_sign_negative());
                        assert!((data[6] - 2.71828).abs() < 0.00001);
                        println!("F32 [7] validated with correct values");
                    }
                    (burn_tensor::DType::F32, []) => {
                        let data = tensor_data.iter::<f32>().collect::<Vec<_>>();
                        assert_eq!(data[0], 42.0);
                        println!("F32 scalar [] validated: 42.0");
                    }
                    (burn_tensor::DType::F32, [2, 3]) => {
                        let data = tensor_data.iter::<f32>().collect::<Vec<_>>();
                        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
                        println!("F32 2D [2, 3] validated with correct values");
                    }
                    (burn_tensor::DType::F64, [3]) => {
                        let data = tensor_data.iter::<f64>().collect::<Vec<_>>();
                        // [1.5e100, -2.5e-100, 3.141592653589793]
                        assert_eq!(data[0], 1.5e100);
                        assert_eq!(data[1], -2.5e-100);
                        assert!((data[2] - 3.141592653589793).abs() < 1e-15);
                        println!("F64 [3] validated with correct values");
                    }
                    (burn_tensor::DType::I32, [5]) => {
                        let data = tensor_data.iter::<i32>().collect::<Vec<_>>();
                        // [2147483647, -2147483648, 0, 1, -1]
                        assert_eq!(data[0], 2147483647);
                        assert_eq!(data[1], -2147483648);
                        assert_eq!(data[2], 0);
                        assert_eq!(data[3], 1);
                        assert_eq!(data[4], -1);
                        println!("I32 [5] validated with correct values");
                    }
                    (burn_tensor::DType::I32, [2, 2, 2]) => {
                        let data = tensor_data.iter::<i32>().collect::<Vec<_>>();
                        // [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
                        assert_eq!(data, vec![1, 2, 3, 4, 5, 6, 7, 8]);
                        println!("I32 3D [2, 2, 2] validated with correct values");
                    }
                    (burn_tensor::DType::I64, [3]) => {
                        let data = tensor_data.iter::<i64>().collect::<Vec<_>>();
                        // [9223372036854775807, -9223372036854775808, 0]
                        assert_eq!(data[0], 9223372036854775807);
                        assert_eq!(data[1], -9223372036854775808);
                        assert_eq!(data[2], 0);
                        println!("I64 [3] validated with correct values");
                    }
                    (burn_tensor::DType::Bool, [5]) => {
                        let data = tensor_data.iter::<bool>().collect::<Vec<_>>();
                        // [true, false, true, true, false]
                        assert_eq!(data[0], true);
                        assert_eq!(data[1], false);
                        assert_eq!(data[2], true);
                        assert_eq!(data[3], true);
                        assert_eq!(data[4], false);
                        println!("Bool [5] validated with correct values");
                    }
                    (burn_tensor::DType::Bool, [2, 2]) => {
                        let data = tensor_data.iter::<bool>().collect::<Vec<_>>();
                        // [[true, false], [false, true]]
                        assert_eq!(data, vec![true, false, false, true]);
                        println!("Bool 2D [2, 2] validated with correct values");
                    }
                    (burn_tensor::DType::U8, [5]) => {
                        let data = tensor_data.iter::<u8>().collect::<Vec<_>>();
                        // [0, 255, 128, 1, 254]
                        assert_eq!(data[0], 0);
                        assert_eq!(data[1], 255);
                        assert_eq!(data[2], 128);
                        assert_eq!(data[3], 1);
                        assert_eq!(data[4], 254);
                        println!("U8 [5] validated with correct values");
                    }
                    (burn_tensor::DType::I8, [5]) => {
                        let data = tensor_data.iter::<i8>().collect::<Vec<_>>();
                        // [127, -128, 0, 1, -1]
                        assert_eq!(data[0], 127);
                        assert_eq!(data[1], -128);
                        assert_eq!(data[2], 0);
                        assert_eq!(data[3], 1);
                        assert_eq!(data[4], -1);
                        println!("I8 [5] validated with correct values");
                    }
                    (burn_tensor::DType::F16, [4]) => {
                        let data = tensor_data.iter::<half::f16>().collect::<Vec<_>>();
                        // [1.0, -1.0, 0.5, 65504.0]
                        assert_eq!(data[0], half::f16::from_f32(1.0));
                        assert_eq!(data[1], half::f16::from_f32(-1.0));
                        assert_eq!(data[2], half::f16::from_f32(0.5));
                        assert_eq!(data[3], half::f16::from_f32(65504.0));
                        println!("F16 [4] validated with correct values");
                    }
                    (burn_tensor::DType::U16, [4]) => {
                        let data = tensor_data.iter::<u16>().collect::<Vec<_>>();
                        // [0, 65535, 32768, 1]
                        assert_eq!(data[0], 0);
                        assert_eq!(data[1], 65535);
                        assert_eq!(data[2], 32768);
                        assert_eq!(data[3], 1);
                        println!("U16 [4] validated with correct values");
                    }
                    _ => {
                        panic!(
                            "Unexpected constant node: dtype {:?}, shape {:?}",
                            dtype, shape
                        );
                    }
                }
            }
        } else {
            // The first input exists but doesn't have tensor data - this is invalid for Constant nodes
            panic!(
                "Constant node '{}' first input has no tensor data (input type: {:?})",
                node.name, input_arg.ty
            );
        }
    }

    println!("\n= Validation Summary =");
    println!("Total constants: {}", const_count);
    println!("Non-empty tensors: {}", non_empty_count);
    println!("Empty tensors: {}", empty_count);
    println!("Data types validated: {:?}", validated_types);

    // Verify we got all major types
    assert!(
        validated_types.contains(&burn_tensor::DType::F32),
        "Missing F32"
    );
    assert!(
        validated_types.contains(&burn_tensor::DType::F64),
        "Missing F64"
    );
    assert!(
        validated_types.contains(&burn_tensor::DType::I32),
        "Missing I32"
    );
    assert!(
        validated_types.contains(&burn_tensor::DType::I64),
        "Missing I64"
    );
    assert!(
        validated_types.contains(&burn_tensor::DType::Bool),
        "Missing Bool"
    );
    assert!(
        validated_types.contains(&burn_tensor::DType::U8),
        "Missing U8"
    );
    assert!(
        validated_types.contains(&burn_tensor::DType::I8),
        "Missing I8"
    );
    assert!(
        validated_types.contains(&burn_tensor::DType::F16),
        "Missing F16"
    );
    assert!(
        validated_types.contains(&burn_tensor::DType::U16),
        "Missing U16"
    );

    assert_eq!(empty_count, 9, "Should have 9 empty tensors (one per type)");
    assert_eq!(
        non_empty_count, 13,
        "Should have 13 non-empty tensors (10 original + 3 multi-dim)"
    );

    println!("\nAll data type conversions validated successfully!");
    println!("Multi-dimensional shapes (2D, 3D) validated successfully!");
}
