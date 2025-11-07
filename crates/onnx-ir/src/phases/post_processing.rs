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
    proto_conversion::MIN_OPSET_VERSION,
};

/// Result of Identity elimination analysis
struct IdentityEliminationPlan {
    /// Mapping from Identity output names to their input names (for rewiring)
    rewire_map: HashMap<String, String>,
    /// Indices of Identity nodes to remove
    nodes_to_remove: HashSet<usize>,
}

/// Rewire an argument to bypass removed Identity nodes
fn rewire_argument(
    arg: &mut Argument,
    rewire_map: &HashMap<String, String>,
    output_arg_map: &HashMap<String, Argument>,
) {
    if let Some(new_name) = rewire_map.get(&arg.name) {
        if let Some(source_arg) = output_arg_map.get(new_name) {
            arg.name = new_name.clone();
            arg.data_id = source_arg.data_id;
            arg.value_store = source_arg.value_store.clone();
            arg.ty = source_arg.ty.clone();
            arg.value_source = source_arg.value_source;
        } else {
            arg.name = new_name.clone();
        }
    }
}

/// Analyze which Identity nodes can be removed and create rewiring map
fn plan_identity_elimination(nodes: &[Node]) -> IdentityEliminationPlan {
    let mut rewire_map = HashMap::new();
    let mut nodes_to_remove = HashSet::new();

    // Find all Identity nodes
    let identity_indices: Vec<usize> = nodes
        .iter()
        .enumerate()
        .filter_map(|(i, node)| (node.node_type == NodeType::Identity).then_some(i))
        .collect();

    // Remove all pass-through Identity nodes (including empty graphs)
    for &idx in &identity_indices {
        let node = &nodes[idx];

        if node.inputs.is_empty() {
            log::warn!("Identity node {} has no inputs, skipping", node.name);
            continue;
        }

        let input_name = &node.inputs[0].name;
        let output_name = &node.outputs[0].name;

        rewire_map.insert(output_name.clone(), input_name.clone());
        nodes_to_remove.insert(idx);
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
        return;
    }

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

    // Step 3: Rewire node inputs
    for node in nodes.iter_mut() {
        for input in &mut node.inputs {
            rewire_argument(input, &resolved_rewire_map, &output_arg_map);
        }
    }

    // Step 4: Rewire graph outputs
    for output in outputs.iter_mut() {
        rewire_argument(output, &resolved_rewire_map, &output_arg_map);
    }

    // Step 5: Remove Identity nodes
    *nodes = nodes
        .drain(..)
        .enumerate()
        .filter_map(|(i, node)| (!nodes_to_remove.contains(&i)).then_some(node))
        .collect();
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

        let result = std::mem::replace(&mut *state, GraphState::new(&[], &[], &[], &[])).consume();

        // Restore tensor_store for .value() access
        state
            .tensor_store
            .restore_data(tensor_data_clone, next_tensor_id);
        result
    };

    // Identity elimination
    log::debug!("Starting Identity elimination");
    {
        let elimination_plan = plan_identity_elimination(&nodes);
        apply_identity_elimination(&mut nodes, &mut outputs, elimination_plan);
    }

    // Re-run constant lifting after identity elimination
    log::debug!("Re-running constant lifting after Identity elimination");
    {
        let mut state = state_rc.borrow_mut();
        state.processed_nodes = nodes.clone();
        drop(state);

        // Re-attach value_store and lift constants
        for node in &mut nodes {
            for arg in &mut node.inputs {
                arg.value_store = Some(state_rc.clone());
            }

            let registry = get_processor_registry();
            let processor = registry.get(&node.node_type);
            // Constant lifting is a best-effort optimization after identity elimination.
            // Not all arguments can be lifted (e.g., already Static, Dynamic), so we log
            // errors but don't fail the pipeline.
            if let Err(e) = processor.lift_constants(node, MIN_OPSET_VERSION as usize) {
                log::debug!(
                    "Could not lift constants for node '{}' (type: {:?}): {:?}",
                    node.name,
                    node.node_type,
                    e
                );
            }
        }
    }

    (nodes, inputs, outputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, Argument, DType, Node, NodeType, TensorType};

    fn create_identity_node(name: &str, input_name: &str, output_name: &str) -> Node {
        Node {
            node_type: NodeType::Identity,
            name: name.to_string(),
            inputs: vec![Argument {
                name: input_name.to_string(),
                ty: ArgType::Tensor(TensorType {
                    dtype: DType::F32,
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
                    dtype: DType::F32,
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
                        dtype: DType::F32,
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
                        dtype: DType::F32,
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
                    dtype: DType::F32,
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

        let plan = plan_identity_elimination(&nodes);

        assert_eq!(plan.nodes_to_remove.len(), 1);
        assert!(plan.nodes_to_remove.contains(&0));
        assert_eq!(
            plan.rewire_map.get("identity1_out"),
            Some(&"input1".to_string())
        );
    }

    #[test]
    fn test_remove_all_identities_in_empty_graph() {
        let nodes = vec![
            create_identity_node("identity1", "input1", "output1"),
            create_identity_node("identity2", "input2", "output2"),
        ];

        let plan = plan_identity_elimination(&nodes);

        // Should remove all Identity nodes (empty graph is allowed)
        assert_eq!(plan.nodes_to_remove.len(), 2);
        assert!(plan.nodes_to_remove.contains(&0));
        assert!(plan.nodes_to_remove.contains(&1));
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
                dtype: DType::F32,
                rank: 2,
                static_shape: None,
            }),
            data_id: None,
            value_source: crate::ir::ValueSource::Dynamic,
            value_store: None,
        }];

        let plan = plan_identity_elimination(&nodes);
        apply_identity_elimination(&mut nodes, &mut outputs, plan);

        // Identity should be removed
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].node_type, NodeType::Add);

        // Add node input should be rewired
        assert_eq!(nodes[0].inputs[0].name, "input1");
    }
}
