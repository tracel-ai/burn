//! Phase 2: Node Conversion
//!
//! Converts ONNX nodes to IR, performs node fusion and remapping:
//!
//! **Coalescing**: Gemm → Linear, MatMul+Add → Linear
//! **Remapping**: Conv → Conv1d/2d/3d, MaxPool → MaxPool1d/2d, etc.

use std::{cell::RefCell, collections::HashMap, iter::Peekable, rc::Rc, slice::Iter};

use crate::{
    graph_state::GraphState,
    ir::{ArgType, AttributeValue, Node, NodeType, TensorData, TensorDataExt},
    pipeline::OnnxIrError,
    processor::get_processor_registry,
    proto_conversion::convert_node_proto,
    protos::{GraphProto, NodeProto},
};

/// Convert all ONNX nodes from GraphProto to IR nodes (for subgraphs)
///
/// # Errors
///
/// Returns an error if graph attribute processing fails
pub(crate) fn convert_nodes_from_graph(
    graph: &GraphProto,
    state_rc: &Rc<RefCell<GraphState>>,
    opset_version: usize,
) -> Result<(), OnnxIrError> {
    convert_nodes_impl(&graph.node, state_rc, opset_version)
}

/// Internal implementation for node conversion
///
/// # Errors
///
/// Returns an error if graph attribute processing fails
fn convert_nodes_impl(
    nodes: &[NodeProto],
    state_rc: &Rc<RefCell<GraphState>>,
    opset_version: usize,
) -> Result<(), OnnxIrError> {
    let mut node_name_counter: HashMap<NodeType, usize> = HashMap::new();

    // Get the name registry (if available)
    let name_registry = {
        let state = state_rc.borrow();

        if let Some(registry) = state.name_registry() {
            // Registry is shared, already initialized with constant count from initializers
            Some(registry.clone())
        } else {
            // Fall back to local counter for backwards compatibility
            let constant_count = state
                .processed_nodes
                .iter()
                .filter(|n| n.node_type == NodeType::Constant)
                .count();
            if constant_count > 0 {
                node_name_counter.insert(NodeType::Constant, constant_count);
            }
            None
        }
    };

    let mut node_iter = nodes.iter().peekable();

    while let Some(node_proto) = node_iter.next() {
        let mut node = convert_node_proto(node_proto, &state_rc.borrow());

        // Handle graph attributes for control flow nodes (If, Loop, Scan)
        if matches!(
            node.node_type,
            NodeType::If | NodeType::Loop | NodeType::Scan
        ) {
            // Pass the current graph's NameRegistry to ensure unique names across nested subgraphs
            // If no registry exists, create one and initialize with current node counts
            let parent_registry = if let Some(registry) = state_rc.borrow().name_registry().cloned()
            {
                registry
            } else {
                // Create new registry and initialize with current node name counters
                let registry = crate::graph_state::NameRegistry::new();
                // Initialize counters from the current graph's already-named nodes
                // IMPORTANT: Increment by 1 to account for the current node which will be
                // renamed later (at line 109) using the local counter
                for (node_type, count) in &node_name_counter {
                    registry.set_initial_counter(node_type, count + 1);
                }
                // Also account for the current node's type
                registry.set_initial_counter(&node.node_type, 1);
                registry
            };

            let graph_attrs = crate::proto_conversion::convert_graph_attributes(
                node_proto,
                opset_version,
                Some(parent_registry),
            );
            // Merge graph attributes with existing attributes
            for (key, value) in graph_attrs {
                node.attrs.insert(key, value);
            }

            // Update subgraph inputs to use renamed names from parent scope
            // This is necessary because onnx-ir renames outputs (e.g., var2 -> add3_out1)
            // but subgraph inputs still reference original ONNX names
            update_subgraph_inputs(&mut node, &state_rc.borrow());
        }

        // Attach value_store to all arguments
        attach_value_stores(&mut node, state_rc);

        // Extract constant from attributes (if Constant node)
        if node.node_type == NodeType::Constant {
            extract_constant_from_attributes(&mut node, state_rc);
        }

        // Remap node types based on patterns
        remap_node_type(&mut node);

        // Rename node with counter
        rename_node(&mut node, &mut node_name_counter, name_registry.as_ref());

        // Track node type before coalesce (may change it)
        let node_type_before = node.node_type.clone();

        // Coalesce with following nodes (fusion)
        coalesce(&mut node, &mut node_iter, &mut state_rc.borrow_mut());

        // Re-attach value_stores after coalesce (may have added inputs)
        attach_value_stores(&mut node, state_rc);

        // Rename if coalesce changed node type
        if node.node_type != node_type_before {
            rename_node(&mut node, &mut node_name_counter, name_registry.as_ref());
        }

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

    Ok(())
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
            AttributeValue::Float32(val) => Some(TensorData::new(vec![*val], vec![])),
            AttributeValue::Float32s(vals) => Some(TensorData::new(vals.clone(), vec![vals.len()])),
            AttributeValue::Int64(val) => Some(TensorData::new(vec![*val], vec![])),
            AttributeValue::Int64s(vals) => Some(TensorData::new(vals.clone(), vec![vals.len()])),
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
                    dtype: tensor_data.elem_type(),
                    rank: tensor_data.shape.len(),
                    static_shape: Some(tensor_data.shape.to_vec()),
                })
            };

            // Create input with Static value
            node.inputs.push(crate::ir::Argument {
                name: String::new(),
                ty: ty.clone(),
                value_source: crate::ir::ValueSource::Static(data_id),
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
fn rename_node(
    node: &mut Node,
    counters: &mut HashMap<NodeType, usize>,
    name_registry: Option<&crate::graph_state::NameRegistry>,
) {
    // If registry is available, use it to generate unique names across subgraphs
    if let Some(registry) = name_registry {
        let old_name = node.name.clone();
        node.name = registry.generate_node_name(&node.node_type);
        log::debug!(
            "Renamed node: '{}' -> '{}' (type: {:?})",
            old_name,
            node.name,
            node.node_type
        );
    } else {
        // Fall back to local counter for backwards compatibility
        counters
            .entry(node.node_type.clone())
            .and_modify(|e| *e += 1)
            .or_insert(1);

        let new_name = format!("{}{}", node.node_type, counters[&node.node_type]).to_lowercase();
        node.name = new_name;
    }
}

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

/// Coalesce adjacent nodes into a single node (Gemm→Linear, MatMul+Add→Linear)
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

/// Convert Gemm to Linear (when alpha=1, beta=1, transB=1)
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
        log::debug!("Fusing Gemm → Linear for node {}", node.name);
        node.node_type = NodeType::Linear;
        node.attrs.remove("alpha");
        node.attrs.remove("beta");
        node.attrs.remove("transB");

        // Transpose the weights
        transpose_linear_node_weights(node, graph_data);
    } else {
        log::debug!(
            "Keeping Gemm node {} (alpha={:?}, beta={:?}, transB={:?} don't match Linear pattern)",
            node.name,
            node.attrs.get("alpha"),
            node.attrs.get("beta"),
            node.attrs.get("transB")
        );
    }
}

/// Transpose linear weights (required for Gemm → Linear conversion)
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
    let data_id = match &node.inputs[1].value_source {
        crate::ir::ValueSource::Static(id) => {
            // Static input with embedded data_id
            *id
        }
        crate::ir::ValueSource::Constant => {
            // Constant input - lookup the constant node to get data_id
            graph_data
                .get_constant_data_id_by_output(&node.inputs[1].name)
                .expect("Constant input must have data_id in constant node")
        }
        _ => panic!("Weight input must be either Static or Constant"),
    };

    let tensor_data = graph_data
        .get_tensor_data(data_id)
        .expect("Weight must have tensor data in central store")
        .clone();

    let shape = tensor_data.shape.clone();

    assert_eq!(shape.len(), 2, "Weight must be a 2D tensor");

    let new_shape = vec![shape[1], shape[0]];

    let new_tensor_data = match tensor_data.elem_type() {
        crate::ir::DType::F32 => {
            let data: Vec<f32> = tensor_data.to_vec().unwrap();
            let data_t = transpose_flattened(data, shape[0], shape[1]);
            TensorData::new(data_t, new_shape)
        }
        crate::ir::DType::F64 => {
            let data: Vec<f64> = tensor_data.to_vec().unwrap();
            let data_t = transpose_flattened(data, shape[0], shape[1]);
            TensorData::new(data_t, new_shape)
        }
        crate::ir::DType::F16 => {
            let data: Vec<half::f16> = tensor_data.to_vec().unwrap();
            let data_t = transpose_flattened(data, shape[0], shape[1]);
            TensorData::new(data_t, new_shape)
        }
        _ => panic!("Only float types are supported for Linear node"),
    };

    // Update the central store with the transposed weights
    if let Some(stored_data) = graph_data.get_tensor_data_mut(data_id) {
        *stored_data = new_tensor_data;
    }

    // Embed the data_id in the input for downstream use (lift_constants may not have run yet)
    node.inputs[1].value_source = crate::ir::ValueSource::Static(data_id);
    node.inputs[1].name.clear(); // Static values are accessed by ID, not name
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

/// Convert MatMul to Linear, fusing the following Add node as bias if present
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
        log::debug!(
            "Keeping MatMul node {} (second input is not a constant weight)",
            node.name
        );
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
    log::debug!("Converting MatMul → Linear for node {}", node.name);

    // Check the next node for potential conversion
    if let Some(peek_node) = iter_mut.peek() {
        let peek_node = convert_node_proto(peek_node, graph_data);
        if is_add_node_with_bias(&peek_node, node, graph_data) {
            convert_and_remove_add_node(&peek_node, node);
            log::debug!("Fused Add bias into Linear node {}", node.name);

            // You don't have to remove it if it's never stored in the first place
            let _ = iter_mut.next();
        }
    }
}

/// Check if the peeked node is an Add with bias for the current MatMul
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

/// Merge the Add node's bias into the MatMul node
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

/// Update subgraph inputs to use renamed names from parent scope
///
/// When onnx-ir processes nodes, it renames outputs (e.g., var2 -> add3_out1).
/// Subgraph inputs reference original ONNX names, so we need to update them
/// to use the renamed names from the parent scope.
fn update_subgraph_inputs(node: &mut Node, graph_state: &GraphState) {
    // Get the node_output_map which maps original ONNX names to (node_idx, output_idx)
    let node_output_map = graph_state.node_output_map();

    // Update inputs for each graph attribute
    for attr_value in node.attrs.values_mut() {
        match attr_value {
            AttributeValue::Graph(subgraph) => {
                update_single_subgraph_inputs(subgraph, node_output_map, graph_state);
            }
            AttributeValue::Graphs(subgraphs) => {
                for subgraph in subgraphs {
                    update_single_subgraph_inputs(subgraph, node_output_map, graph_state);
                }
            }
            _ => {}
        }
    }
}

/// Update inputs for a single subgraph
fn update_single_subgraph_inputs(
    subgraph: &mut crate::ir::OnnxGraph,
    node_output_map: &HashMap<String, (usize, usize)>,
    graph_state: &GraphState,
) {
    // Build a mapping of old names to new names
    let mut rename_map = HashMap::new();

    for input in &mut subgraph.inputs {
        // Check if this input name exists in the parent's node_output_map
        if let Some(&(node_idx, output_idx)) = node_output_map.get(&input.name) {
            // Get the renamed output name from the parent node
            let renamed_name = graph_state.processed_nodes[node_idx].outputs[output_idx]
                .name
                .clone();
            log::debug!(
                "Updating subgraph input: {} -> {}",
                input.name,
                renamed_name
            );

            // Store the mapping
            rename_map.insert(input.name.clone(), renamed_name.clone());

            // Update the input name
            input.name = renamed_name;
        }
    }

    // Also update all node inputs within the subgraph that reference the old names
    for node in &mut subgraph.nodes {
        for node_input in &mut node.inputs {
            if let Some(new_name) = rename_map.get(&node_input.name) {
                log::debug!(
                    "Updating node input in subgraph: {} -> {}",
                    node_input.name,
                    new_name
                );
                node_input.name = new_name.clone();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::test_utils::NodeBuilder;

    #[test]
    fn should_infer_conv2d_node_from_weights_rank() {
        // Weight tensor data - not important for the test
        let weight_data = vec![0.0; 16];
        // [out_channels, in_channels, k_h, k_w] = [2, 2, 2, 2] = 16 elements
        let weight_shape = vec![2, 2, 2, 2];

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
