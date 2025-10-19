//! Phase 2: Node Conversion
//!
//! Converts ONNX nodes to IR nodes, extracts constants from attributes,
//! performs coalescing, and lifts constants.
//!
//! ## Node Coalescing
//!
//! This phase coalesces multiple ONNX nodes into single IR nodes:
//! - Gemm → Linear (when alpha=1, beta=1, transB=1)
//! - MatMul + Add → Linear (with bias)
//!
//! ## Node Type Remapping
//!
//! This phase remaps generic node types to dimensional-specific versions:
//! - Conv → Conv1d/Conv2d/Conv3d (based on kernel_shape)
//! - ConvTranspose → ConvTranspose1d/ConvTranspose2d/ConvTranspose3d
//! - MaxPool → MaxPool1d/MaxPool2d
//! - AveragePool → AveragePool1d/AveragePool2d

use std::{cell::RefCell, collections::HashMap, iter::Peekable, rc::Rc, slice::Iter};

use crate::{
    graph_state::GraphState,
    ir::{ArgType, AttributeValue, Data, Node, NodeType, TensorData},
    processor::get_processor_registry,
    proto_conversion::convert_node_proto,
    protos::{ModelProto, NodeProto},
};

/// Convert all ONNX nodes to IR nodes
pub(crate) fn convert_nodes(model: &ModelProto, state_rc: &Rc<RefCell<GraphState>>) {
    let opset_version = extract_opset_version(model);
    let mut node_name_counter: HashMap<NodeType, usize> = HashMap::new();

    // Initialize constant counter from initializers
    {
        let state = state_rc.borrow();
        let constant_count = state
            .processed_nodes
            .iter()
            .filter(|n| n.node_type == NodeType::Constant)
            .count();
        if constant_count > 0 {
            node_name_counter.insert(NodeType::Constant, constant_count);
            log::debug!(
                "Initialized Constant counter to {} (from initializers)",
                constant_count
            );
        }
    }

    let mut node_iter = model.graph.node.iter().peekable();

    while let Some(node_proto) = node_iter.next() {
        let mut node = convert_node_proto(node_proto, &state_rc.borrow());

        // Attach value_store to all arguments
        attach_value_stores(&mut node, state_rc);

        // Extract constant from attributes (if Constant node)
        if node.node_type == NodeType::Constant {
            extract_constant_from_attributes(&mut node, state_rc);
        }

        // Remap node types based on patterns
        remap_node_type(&mut node);

        // Rename node with counter
        rename_node(&mut node, &mut node_name_counter);

        // Track node type before coalesce (may change it)
        let node_type_before = node.node_type.clone();

        // Coalesce with following nodes (fusion)
        coalesce(&mut node, &mut node_iter, &mut state_rc.borrow_mut());

        // Re-attach value_stores after coalesce (may have added inputs)
        attach_value_stores(&mut node, state_rc);

        // Rename if coalesce changed node type
        if node.node_type != node_type_before {
            rename_node(&mut node, &mut node_name_counter);
        }

        log::debug!("Processing node: {}", node.name);

        // Lift constants and extract config
        let registry = get_processor_registry();
        let processor = registry.get(&node.node_type);

        processor
            .lift_constants(&mut node, opset_version)
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to lift constants for node {} (type: {:?}): {:?}",
                    node.name, node.node_type, e
                )
            });

        let config = processor
            .extract_config(&node, opset_version)
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to extract config for node {} (type: {:?}): {:?}",
                    node.name, node.node_type, e
                )
            });
        node.config = config;

        // Add to graph state
        state_rc.borrow_mut().add_node(node);
    }

    log::debug!("Converted {} ONNX nodes", model.graph.node.len());
}

/// Extract constant data from node attributes and move to tensor store
fn extract_constant_from_attributes(node: &mut Node, state_rc: &Rc<RefCell<GraphState>>) {
    let keys = [
        "value",
        "value_float",
        "value_floats",
        "value_int",
        "value_ints",
        "value_string",
        "value_strings",
    ];

    if let Some(attr_key) = keys.iter().find(|&key| node.attrs.contains_key(*key))
        && let Some(attr_value) = node.attrs.get(*attr_key)
    {
        let tensor_data_opt: Option<TensorData> = match attr_value {
            AttributeValue::Tensor(tensor) => Some(tensor.clone()),
            AttributeValue::Float32(val) => Some(TensorData {
                shape: vec![],
                data: crate::ir::Data::Float32(*val),
            }),
            AttributeValue::Float32s(vals) => Some(TensorData {
                shape: vec![vals.len()],
                data: crate::ir::Data::Float32s(vals.clone()),
            }),
            AttributeValue::Int64(val) => Some(TensorData {
                shape: vec![],
                data: crate::ir::Data::Int64(*val),
            }),
            AttributeValue::Int64s(vals) => Some(TensorData {
                shape: vec![vals.len()],
                data: crate::ir::Data::Int64s(vals.clone()),
            }),
            _ => None,
        };

        if let Some(tensor_data) = tensor_data_opt {
            // Store in central tensor store
            let data_id = {
                let mut state = state_rc.borrow_mut();
                state.store_tensor_data(tensor_data.clone())
            };

            // Create type from tensor data
            let ty = if tensor_data.shape.is_empty() {
                crate::ir::ArgType::Scalar(tensor_data.elem_type())
            } else {
                crate::ir::ArgType::Tensor(crate::ir::TensorType {
                    elem_type: tensor_data.elem_type(),
                    rank: tensor_data.shape.len(),
                    static_shape: Some(tensor_data.shape.clone()),
                })
            };

            // Create input with Static value
            node.inputs.push(crate::ir::Argument {
                name: String::new(),
                ty: ty.clone(),
                data_id: Some(data_id),
                value_source: crate::ir::ValueSource::Static,
                value_store: Some(state_rc.clone()),
            });

            // Set output type and value_source
            if !node.outputs.is_empty() {
                node.outputs[0].value_source = crate::ir::ValueSource::Constant;
                node.outputs[0].ty = ty;
            }

            // Remove from attributes
            node.attrs.remove(*attr_key);
        }
    }
}

/// Attach value_store references to all node arguments
fn attach_value_stores(node: &mut Node, state_rc: &Rc<RefCell<GraphState>>) {
    for arg in &mut node.inputs {
        arg.value_store = Some(state_rc.clone());
    }
    for arg in &mut node.outputs {
        arg.value_store = Some(state_rc.clone());
    }
}

/// Rename node with type-based counter
fn rename_node(node: &mut Node, counters: &mut HashMap<NodeType, usize>) {
    counters
        .entry(node.node_type.clone())
        .and_modify(|e| *e += 1)
        .or_insert(1);

    let new_name = format!("{}{}", node.node_type, counters[&node.node_type]).to_lowercase();
    log::debug!("Renaming node {:?} to {new_name:?}", &node.name);
    node.name = new_name;
}

/// Extract opset version from model
fn extract_opset_version(model: &ModelProto) -> usize {
    model
        .opset_import
        .iter()
        .find(|opset| opset.domain.is_empty())
        .map(|opset| opset.version as usize)
        .unwrap_or(17) // MIN_OPSET_VERSION
}

// ============================================================================
// Node Type Remapping
// ============================================================================

/// Remap node type using kernel shape
fn remap_node_with_kernel_shape<F>(node: &mut Node, new_node_type: F)
where
    F: FnOnce(usize) -> NodeType,
{
    let spatial_dims = match node.attrs.get("kernel_shape") {
        Some(AttributeValue::Int64s(ints)) => ints.len(),
        None if [NodeType::Conv, NodeType::ConvTranspose].contains(&node.node_type) => {
            // "kernel_shape" attribute is optional and should be inferred from weights
            // https://onnx.ai/onnx/operators/onnx__Conv.html
            if let ArgType::Tensor(weight) = &node.inputs[1].ty {
                // Skip leading channels in/out
                weight.rank - 2
            } else {
                panic!("Cannot infer kernel spatial dims");
            }
        }
        _ => panic!("Cannot infer kernel shape"),
    };
    node.node_type = new_node_type(spatial_dims);
}

/// Remap node type to a more specific one
fn remap_node_type(node: &mut Node) {
    match node.node_type {
        NodeType::Conv => remap_node_with_kernel_shape(node, |spatial_dims| match spatial_dims {
            1 => NodeType::Conv1d,
            2 => NodeType::Conv2d,
            3 => NodeType::Conv3d,
            _ => panic!("Only conv 1d, 2d and 3d are supported"),
        }),
        NodeType::ConvTranspose => {
            remap_node_with_kernel_shape(node, |spatial_dims| match spatial_dims {
                1 => NodeType::ConvTranspose1d,
                2 => NodeType::ConvTranspose2d,
                3 => NodeType::ConvTranspose3d,
                _ => panic!("Only conv_transpose 1d, 2d and 3d are supported"),
            })
        }
        NodeType::MaxPool => {
            remap_node_with_kernel_shape(node, |spatial_dims| match spatial_dims {
                1 => NodeType::MaxPool1d,
                2 => NodeType::MaxPool2d,
                _ => panic!("Only max_pool 1d and 2d are supported"),
            })
        }
        NodeType::AveragePool => {
            remap_node_with_kernel_shape(node, |spatial_dims| match spatial_dims {
                1 => NodeType::AveragePool1d,
                2 => NodeType::AveragePool2d,
                _ => panic!("Only avg_pool 1d and 2d are supported"),
            })
        }
        _ => (),
    }
}

// ============================================================================
// Node Coalescing
// ============================================================================

/// The function transforms the graph into a new one where the nodes are coalesced into a single node.
fn coalesce(
    node: &mut Node,
    nodes_iter: &mut Peekable<Iter<NodeProto>>,
    graph_data: &mut GraphState,
) {
    #[allow(clippy::single_match)]
    match node.node_type {
        NodeType::Gemm => convert_gemm_to_linear(node, graph_data),
        NodeType::MatMul => {
            convert_matmul_to_linear(node, nodes_iter, graph_data);
        }
        _ => {}
    }
}

/// This function converts a Gemm node into a Linear node
///
/// PyTorch and other frameworks use Gemm node to represent Linear layer.
fn convert_gemm_to_linear(node: &mut Node, graph_data: &mut GraphState) {
    if node.outputs.len() != 1 {
        panic!("Gemm node must have 1 output");
    }
    let straight_linear = match (
        node.attrs.get("alpha"),
        node.attrs.get("beta"),
        node.attrs.get("transB"),
    ) {
        (
            Some(AttributeValue::Float32(alpha)),
            Some(AttributeValue::Float32(beta)),
            Some(AttributeValue::Int64(trans_b)),
        ) => *alpha == 1.0 && *beta == 1.0 && *trans_b == 1,
        _ => false,
    };

    if straight_linear {
        node.node_type = NodeType::Linear;
        node.attrs.remove("alpha");
        node.attrs.remove("beta");
        node.attrs.remove("transB");

        // Transpose the weights
        transpose_linear_node_weights(node, graph_data);
    }
}

// Transpose linear weights (required for Gemm -> Linear conversion)
fn transpose_linear_node_weights(node: &mut Node, graph_data: &mut GraphState) {
    assert!(
        node.inputs.len() > 1,
        "Linear node must have at least 2 input"
    );

    assert!(
        graph_data.has_value(&node.inputs[1].name),
        "Input must have a value"
    );

    // Get the data_id - either directly from Static input, or lookup from Constant input
    let data_id = if let Some(id) = node.inputs[1].data_id {
        // Static input with embedded data_id
        id
    } else if node.inputs[1].is_constant() {
        // Constant input - lookup the constant node to get data_id
        graph_data
            .get_constant_data_id_by_output(&node.inputs[1].name)
            .expect("Constant input must have data_id in constant node")
    } else {
        panic!("Weight input must be either Static or Constant");
    };

    let tensor_data = graph_data
        .get_tensor_data(data_id)
        .expect("Weight must have tensor data in central store")
        .clone();

    let data = &tensor_data.data;
    let shape = &tensor_data.shape;

    assert_eq!(shape.len(), 2, "Weight must be a 2D tensor");

    let new_shape = vec![shape[1], shape[0]];

    let new_tensor_data = match data {
        Data::Float32s(data) => {
            let data_t = transpose_flattened(data.clone(), shape[0], shape[1]);
            TensorData {
                data: Data::Float32s(data_t),
                shape: new_shape,
            }
        }
        Data::Float64s(data) => {
            let data_t = transpose_flattened(data.clone(), shape[0], shape[1]);
            TensorData {
                data: Data::Float64s(data_t),
                shape: new_shape,
            }
        }
        Data::Float16s(data) => {
            let data_t = transpose_flattened(data.clone(), shape[0], shape[1]);
            TensorData {
                data: Data::Float16s(data_t),
                shape: new_shape,
            }
        }
        _ => panic!("Only float types are supported for Linear node"),
    };

    // Update the central store with the transposed weights
    if let Some(stored_data) = graph_data.get_tensor_data_mut(data_id) {
        *stored_data = new_tensor_data;
    }

    // Embed the data_id in the input for downstream use (lift_constants may not have run yet)
    node.inputs[1].data_id = Some(data_id);
}

fn transpose_flattened<T: Copy>(matrix: Vec<T>, rows: usize, cols: usize) -> Vec<T> {
    assert_eq!(matrix.len(), rows * cols, "Matrix must be flattened");

    let mut transposed: Vec<T> = vec![matrix[0]; matrix.len()];

    for i in 0..rows {
        for j in 0..cols {
            transposed[j * rows + i] = matrix[i * cols + j];
        }
    }

    transposed
}

/// This function converts a MatMul node into a Linear node if possible.
///
/// PyTorch and other frameworks use MatMul node to represent Linear layer.
///
/// This function also converts the following Add node into a Linear node if possible.
/// Add node is used to represent bias in PyTorch.
fn convert_matmul_to_linear(
    node: &mut Node,
    iter_mut: &mut Peekable<Iter<NodeProto>>,
    graph_data: &mut GraphState,
) {
    if node.inputs.len() != 2 {
        panic!("MatMul node must have 2 inputs");
    }

    // if the second input does not have a value, it is not a weight, then proceed to the next node
    if !graph_data.has_value(&node.inputs[1].name) {
        return;
    }

    // Check if the second input is a 2D tensor
    if let ArgType::Tensor(ref tensor_type) = node.inputs[1].ty {
        assert_eq!(tensor_type.rank, 2, "Weight must be a 2D tensor");
    } else {
        panic!("Tensor input is expected");
    }

    // Convert the node to Linear
    node.node_type = NodeType::Linear;

    log::debug!("peeking next node for bias conversion");
    // Check the next node for potential conversion
    if let Some(peek_node) = iter_mut.peek() {
        let peek_node = convert_node_proto(peek_node, graph_data);
        if is_add_node_with_bias(&peek_node, node, graph_data) {
            convert_and_remove_add_node(&peek_node, node);

            // You don't have to remove it if it's never stored in the first place
            let _ = iter_mut.next();
        }
    }
}

/// Helper function to check if the peeked node is an Add node with bias
fn is_add_node_with_bias(peek_node: &Node, current_node: &Node, graph_data: &GraphState) -> bool {
    // Check structural requirements first
    if peek_node.node_type != NodeType::Add || peek_node.inputs.len() != 2 {
        return false;
    }

    // Check if one input is the matmul output and the other has a value
    (peek_node.inputs[0].name == current_node.outputs[0].name
        && graph_data.has_value(&peek_node.inputs[1].name))
        || (peek_node.inputs[1].name == current_node.outputs[0].name
            && graph_data.has_value(&peek_node.inputs[0].name))
}

/// Helper function to convert and remove the Add node
fn convert_and_remove_add_node(bias_node: &Node, current_node: &mut Node) {
    // The bias is whichever input is NOT the matmul output
    let bias_input = if bias_node.inputs[0].name == current_node.outputs[0].name {
        bias_node.inputs[1].clone()
    } else {
        bias_node.inputs[0].clone()
    };

    // Push the bias input and update the output name
    current_node.inputs.push(bias_input);
    current_node.outputs[0]
        .name
        .clone_from(&bias_node.outputs[0].name);
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::test_utils::NodeBuilder;

    #[test]
    fn should_infer_conv2d_node_from_weights_rank() {
        // Weight tensor data - not important for the test
        let weight_data = vec![0.0; 16];
        // [.., k_h, k_w]
        let weight_shape = vec![4, 2, 2, 2];

        let mut node = NodeBuilder::new(NodeType::Conv, "test_conv2d")
            .input_tensor_f32("data", 4, None)
            .input_tensor_f32_data("weight", weight_data.clone(), weight_shape)
            .output_tensor_f32("output", 4, None)
            // .attr_ints("kernel_shape", kernel_shape)
            .attr_ints("strides", vec![1, 1])
            .attr_ints("pads", vec![0, 0, 0, 0])
            .attr_ints("dilations", vec![1, 1])
            .attr_int("group", 1)
            .build();

        assert_eq!(node.node_type, NodeType::Conv);
        remap_node_type(&mut node);
        assert_eq!(node.node_type, NodeType::Conv2d);
    }

    #[test]
    fn should_infer_conv_transpose1d_node_from_weights_rank() {
        // Weight tensor data - not important for the test
        let weight_data = vec![0.0; 16];
        // [.., kernel_size]
        let weight_shape = vec![2, 2, 4];

        let mut node = NodeBuilder::new(NodeType::ConvTranspose, "test_conv2d")
            .input_tensor_f32("data", 3, None)
            .input_tensor_f32_data("weight", weight_data, weight_shape)
            .output_tensor_f32("output", 3, None)
            .build();

        assert_eq!(node.node_type, NodeType::ConvTranspose);
        remap_node_type(&mut node);
        assert_eq!(node.node_type, NodeType::ConvTranspose1d);
    }
}
