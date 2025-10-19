//! Phase 4: Post-processing
//!
//! Eliminates Identity nodes by rewiring consumers, preserves at least one if graph would be empty.

use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    rc::Rc,
};

use crate::{
    graph_state::GraphState,
    ir::{Argument, Node, NodeType},
    processor::get_processor_registry,
};

/// Result of Identity elimination analysis
struct IdentityEliminationPlan {
    /// Mapping from Identity output names to their input names (for rewiring)
    rewire_map: HashMap<String, String>,
    /// Indices of Identity nodes to remove
    nodes_to_remove: HashSet<usize>,
}

/// Analyze which Identity nodes can be removed and create rewiring map
fn plan_identity_elimination(
    nodes: &[Node],
    graph_inputs: &[Argument],
    graph_outputs: &[Argument],
) -> IdentityEliminationPlan {
    let mut rewire_map = HashMap::new();
    let mut nodes_to_remove = HashSet::new();

    // Find all non-Constant Identity nodes (Constant Identities were already converted)
    let identity_indices: Vec<usize> = nodes
        .iter()
        .enumerate()
        .filter_map(|(i, node)| {
            if node.node_type == NodeType::Identity {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    // Count non-Identity nodes
    let non_identity_count = nodes.len() - identity_indices.len();

    // Edge case: If graph would be empty after removing all Identities,
    // preserve at least one Identity node
    if non_identity_count == 0 && !identity_indices.is_empty() {
        log::debug!(
            "Graph has only Identity nodes ({}), preserving first Identity to avoid empty graph",
            identity_indices.len()
        );

        // Preserve the first Identity, remove the rest
        for &idx in identity_indices.iter().skip(1) {
            let node = &nodes[idx];
            if !node.inputs.is_empty() {
                let output_name = &node.outputs[0].name;
                let input_name = &node.inputs[0].name;
                rewire_map.insert(output_name.clone(), input_name.clone());
                nodes_to_remove.insert(idx);
                log::debug!(
                    "Removing Identity node {} (rewiring {} -> {})",
                    node.name,
                    output_name,
                    input_name
                );
            }
        }

        return IdentityEliminationPlan {
            rewire_map,
            nodes_to_remove,
        };
    }

    // Check if any graph input directly connects to graph output through Identity
    let mut preserve_identity_for_direct_io = false;
    let graph_input_names: HashSet<String> =
        graph_inputs.iter().map(|arg| arg.name.clone()).collect();
    let graph_output_names: HashSet<String> =
        graph_outputs.iter().map(|arg| arg.name.clone()).collect();

    // Normal case: Remove all pass-through Identity nodes
    for &idx in &identity_indices {
        let node = &nodes[idx];

        if node.inputs.is_empty() {
            log::warn!(
                "Identity node {} has no inputs, cannot eliminate",
                node.name
            );
            continue;
        }

        let input_name = &node.inputs[0].name;
        let output_name = &node.outputs[0].name;

        // Check if this Identity connects a graph input to graph output
        let is_direct_io_connection =
            graph_input_names.contains(input_name) && graph_output_names.contains(output_name);

        // If this is a direct input->output connection and graph has other nodes,
        // preserve one such Identity
        if is_direct_io_connection && non_identity_count > 0 && !preserve_identity_for_direct_io {
            preserve_identity_for_direct_io = true;
            log::debug!(
                "Preserving Identity node {} for direct input->output connection ({} -> {})",
                node.name,
                input_name,
                output_name
            );
            continue;
        }

        // Mark for removal and add to rewire map
        rewire_map.insert(output_name.clone(), input_name.clone());
        nodes_to_remove.insert(idx);
        log::debug!(
            "Removing Identity node {} (rewiring {} -> {})",
            node.name,
            output_name,
            input_name
        );
    }

    IdentityEliminationPlan {
        rewire_map,
        nodes_to_remove,
    }
}

/// Apply the identity elimination plan to the graph
///
/// This function:
/// 1. Rewires all node inputs to bypass removed Identity nodes
/// 2. Updates graph outputs to bypass removed Identity nodes
/// 3. Filters out removed nodes
fn apply_identity_elimination(
    nodes: &mut Vec<Node>,
    outputs: &mut [Argument],
    plan: IdentityEliminationPlan,
) {
    let IdentityEliminationPlan {
        rewire_map,
        nodes_to_remove,
    } = plan;

    if nodes_to_remove.is_empty() {
        log::debug!("No Identity nodes to remove");
        return;
    }

    log::debug!(
        "Applying Identity elimination: removing {} nodes, rewiring {} connections",
        nodes_to_remove.len(),
        rewire_map.len()
    );

    // Step 1: Build a map from output names to Arguments
    // This allows us to look up data_id and value_store when rewiring
    let mut output_arg_map: HashMap<String, Argument> = HashMap::new();
    for node in nodes.iter() {
        for output in &node.outputs {
            output_arg_map.insert(output.name.clone(), output.clone());
        }
    }

    // Step 2: Resolve transitive rewiring
    // If A -> B and B -> C, we need to make sure we map A -> C directly
    let mut resolved_rewire_map = rewire_map.clone();
    for (output, input) in rewire_map.iter() {
        let mut current = input.clone();
        let mut visited = HashSet::new();
        visited.insert(output.clone());

        // Follow the chain until we reach a non-rewired input
        while let Some(next) = resolved_rewire_map.get(&current) {
            if visited.contains(next) {
                // Cycle detected, break
                log::warn!("Cycle detected in rewiring: {:?}", visited);
                break;
            }
            visited.insert(current.clone());
            current = next.clone();
        }

        resolved_rewire_map.insert(output.clone(), current);
    }

    // Step 3: Rewire all node inputs to bypass removed Identity nodes
    // Copy data_id, value_store, and ty from source argument
    for node in nodes.iter_mut() {
        for input in &mut node.inputs {
            if let Some(original_input_name) = resolved_rewire_map.get(&input.name) {
                log::debug!(
                    "Rewiring input {} -> {} in node {}",
                    input.name,
                    original_input_name,
                    node.name
                );

                // Look up the source argument to copy its data_id, value_store, and value_source
                if let Some(source_arg) = output_arg_map.get(original_input_name) {
                    input.name = original_input_name.clone();
                    input.data_id = source_arg.data_id;
                    input.value_store = source_arg.value_store.clone();
                    input.ty = source_arg.ty.clone();
                    input.value_source = source_arg.value_source;
                } else {
                    // Fallback: just update the name if source not found
                    input.name = original_input_name.clone();
                }
            }
        }
    }

    // Step 4: Update graph outputs to bypass removed Identity nodes
    // Copy data_id, value_store, and ty from source argument
    for output in outputs.iter_mut() {
        if let Some(original_output_name) = resolved_rewire_map.get(&output.name) {
            log::debug!(
                "Rewiring graph output {} -> {}",
                output.name,
                original_output_name
            );

            // Look up the source argument to copy its data_id, value_store, and value_source
            if let Some(source_arg) = output_arg_map.get(original_output_name) {
                output.name = original_output_name.clone();
                output.data_id = source_arg.data_id;
                output.value_store = source_arg.value_store.clone();
                output.ty = source_arg.ty.clone();
                output.value_source = source_arg.value_source;
            } else {
                // Fallback: just update the name if source not found
                output.name = original_output_name.clone();
            }
        }
    }

    // Step 5: Remove the Identity nodes
    let mut i = 0;
    nodes.retain(|_| {
        let keep = !nodes_to_remove.contains(&i);
        i += 1;
        keep
    });

    log::debug!("After Identity elimination: {} nodes remain", nodes.len());
}

/// Post-process the graph: eliminate identities and re-lift constants
///
/// Returns (nodes, inputs, outputs) tuple ready for finalization
pub(crate) fn post_process(
    state_rc: &Rc<RefCell<GraphState>>,
) -> (Vec<Node>, Vec<Argument>, Vec<Argument>) {
    // Extract graph data while preserving tensor_store
    let (mut nodes, inputs, mut outputs) = {
        let mut state = state_rc.borrow_mut();
        let tensor_data_clone = state.tensor_store.clone_data();
        let next_tensor_id = state.tensor_store.next_id();

        let result = std::mem::replace(&mut *state, GraphState::new(&[], &[], &[])).consume();

        // Restore tensor_store for .value() access
        state
            .tensor_store
            .restore_data(tensor_data_clone, next_tensor_id);
        result
    };

    // Identity elimination
    log::debug!("Starting Identity elimination");
    {
        let elimination_plan = plan_identity_elimination(&nodes, &inputs, &outputs);
        apply_identity_elimination(&mut nodes, &mut outputs, elimination_plan);
    }
    log::debug!("After Identity elimination: {} nodes remain", nodes.len());

    // Re-run constant lifting after identity elimination
    log::debug!("Re-running constant lifting after Identity elimination");
    {
        let mut state = state_rc.borrow_mut();
        state.processed_nodes = nodes.clone();

        // Rebuild constant_nodes map
        let mut new_constant_nodes = std::collections::HashMap::new();
        for (idx, n) in nodes.iter().enumerate() {
            if n.node_type == NodeType::Constant {
                for output in &n.outputs {
                    new_constant_nodes.insert(output.name.clone(), idx);
                }
            }
        }
        state.constant_nodes = new_constant_nodes;
        drop(state);

        // Re-attach value_store and lift constants
        for node in &mut nodes {
            for arg in &mut node.inputs {
                arg.value_store = Some(state_rc.clone());
            }

            let registry = get_processor_registry();
            let processor = registry.get(&node.node_type);
            let _ = processor.lift_constants(node, 17); // Ignore errors - already lifted
        }
    }

    (nodes, inputs, outputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, Argument, ElementType, Node, NodeType, TensorType};

    fn create_identity_node(name: &str, input_name: &str, output_name: &str) -> Node {
        Node {
            node_type: NodeType::Identity,
            name: name.to_string(),
            inputs: vec![Argument {
                name: input_name.to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 2,
                    static_shape: None,
                }),
                data_id: None,
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            outputs: vec![Argument {
                name: output_name.to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 2,
                    static_shape: None,
                }),
                data_id: None,
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            attrs: Default::default(),
            config: None,
        }
    }

    fn create_add_node(name: &str, input1: &str, input2: &str, output: &str) -> Node {
        Node {
            node_type: NodeType::Add,
            name: name.to_string(),
            inputs: vec![
                Argument {
                    name: input1.to_string(),
                    ty: ArgType::Tensor(TensorType {
                        elem_type: ElementType::Float32,
                        rank: 2,
                        static_shape: None,
                    }),
                    data_id: None,
                    value_source: crate::ir::ValueSource::Dynamic,
                    value_store: None,
                },
                Argument {
                    name: input2.to_string(),
                    ty: ArgType::Tensor(TensorType {
                        elem_type: ElementType::Float32,
                        rank: 2,
                        static_shape: None,
                    }),
                    data_id: None,
                    value_source: crate::ir::ValueSource::Dynamic,
                    value_store: None,
                },
            ],
            outputs: vec![Argument {
                name: output.to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 2,
                    static_shape: None,
                }),
                data_id: None,
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            attrs: Default::default(),
            config: None,
        }
    }

    #[test]
    fn test_remove_single_identity() {
        let nodes = vec![
            create_identity_node("identity1", "input1", "identity1_out"),
            create_add_node("add1", "identity1_out", "input2", "output1"),
        ];

        let plan = plan_identity_elimination(&nodes, &[], &[]);

        assert_eq!(plan.nodes_to_remove.len(), 1);
        assert!(plan.nodes_to_remove.contains(&0));
        assert_eq!(
            plan.rewire_map.get("identity1_out"),
            Some(&"input1".to_string())
        );
    }

    #[test]
    fn test_preserve_identity_for_empty_graph() {
        let nodes = vec![
            create_identity_node("identity1", "input1", "output1"),
            create_identity_node("identity2", "input2", "output2"),
        ];

        let plan = plan_identity_elimination(&nodes, &[], &[]);

        // Should preserve first Identity, remove second
        assert_eq!(plan.nodes_to_remove.len(), 1);
        assert!(plan.nodes_to_remove.contains(&1));
        assert!(!plan.nodes_to_remove.contains(&0));
    }

    #[test]
    fn test_apply_identity_elimination() {
        let mut nodes = vec![
            create_identity_node("identity1", "input1", "identity1_out"),
            create_add_node("add1", "identity1_out", "input2", "add1_out"),
        ];

        let mut outputs = vec![Argument {
            name: "add1_out".to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                rank: 2,
                static_shape: None,
            }),
            data_id: None,
            value_source: crate::ir::ValueSource::Dynamic,
            value_store: None,
        }];

        let plan = plan_identity_elimination(&nodes, &[], &outputs);
        apply_identity_elimination(&mut nodes, &mut outputs, plan);

        // Identity should be removed
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].node_type, NodeType::Add);

        // Add node input should be rewired
        assert_eq!(nodes[0].inputs[0].name, "input1");
    }
}
