//! Phase 3: Type Inference
//!
//! Iterative type inference with preference propagation until convergence.

use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    rc::Rc,
};

use crate::{
    graph_state::GraphState,
    ir::{ArgType, RawNode},
    processor::{ArgPreference, ProcessError, get_processor_registry},
};

/// Infer types for all nodes (extracts nodes to avoid borrow conflicts)
pub(crate) fn infer_types(
    state_rc: &Rc<RefCell<GraphState>>,
    opset_version: usize,
) -> Result<(), ProcessError> {
    // Extract nodes temporarily to avoid holding mutable borrow during type inference
    // (type inference may call .value() which needs immutable borrows)
    let mut nodes = std::mem::take(&mut state_rc.borrow_mut().processed_nodes);
    iterative_type_inference_with_preferences(&mut nodes, opset_version)?;
    state_rc.borrow_mut().processed_nodes = nodes;
    Ok(())
}

/// Iterative type inference with preference propagation
///
/// Algorithm: Build preferences → Sync types → Infer → Collect new preferences → Check convergence
///
/// This allows runtime preference collection (e.g., Concat requests Shape after seeing Shape inputs).
pub(super) fn iterative_type_inference_with_preferences(
    nodes: &mut [RawNode],
    opset: usize,
) -> Result<(), ProcessError> {
    let registry = get_processor_registry();

    // Track collected preferences: (producer_output_name, consumer_name, pref_type_str)
    let mut collected_preferences: HashSet<(String, String, String)> = HashSet::new();

    let max_iterations = 10; // Safety limit to prevent infinite loops

    for iteration in 1..=max_iterations {
        // Step 1: Build OutputPreferences map from collected preferences
        let mut node_preferences: HashMap<String, crate::processor::OutputPreferences> =
            HashMap::new();

        for (output_name, consumer_name, pref_type_str) in &collected_preferences {
            let pref = match pref_type_str.as_str() {
                "Scalar" => ArgPreference::Scalar,
                "Shape" => ArgPreference::Shape,
                "Tensor" => ArgPreference::Tensor,
                _ => continue,
            };

            // Find producer node for this output
            for node in nodes.iter() {
                if node.outputs.iter().any(|o| &o.name == output_name) {
                    node_preferences.entry(node.name.clone()).or_default().add(
                        output_name.clone(),
                        consumer_name.clone(),
                        pref,
                    );
                    break;
                }
            }
        }

        // Step 2: Sync input types from producer outputs (skipped on iteration 1)
        // Iteration 1: Let nodes infer from proto defaults first
        // Iteration 2+: Pre-sync ensures nodes see inferred types, not stale defaults
        if iteration > 1 {
            let output_types: HashMap<String, ArgType> = nodes
                .iter()
                .flat_map(|n| n.outputs.iter().map(|o| (o.name.clone(), o.ty.clone())))
                .collect();

            for node in nodes.iter_mut() {
                for input in &mut node.inputs {
                    if let Some(new_type) = output_types.get(&input.name) {
                        input.ty = new_type.clone();
                    }
                }
            }
        }

        // Step 3: Run infer_types on all nodes with current preferences
        // AND sync types after each node to allow downstream nodes to see updated types
        // within the same iteration (intra-iteration propagation)
        let mut types_changed = false;

        for i in 0..nodes.len() {
            // Get preferences for this node
            let prefs = node_preferences
                .get(&nodes[i].name)
                .cloned()
                .unwrap_or_else(crate::processor::OutputPreferences::new);

            // Validate node against its spec before processing
            let processor = registry.get(&nodes[i].node_type);
            let spec = processor.spec();
            crate::processor::validate_node_spec(&nodes[i], opset, &spec)?;

            // Run type inference on this node
            processor.infer_types(&mut nodes[i], opset, &prefs)?;

            // Validate that no rank-0 tensors were created (invariant: should be Scalars)
            crate::processor::validate_no_rank_zero_tensors(&nodes[i])?;

            // Immediately sync this node's output types to downstream nodes' inputs
            // This allows downstream nodes to see correct types in the same iteration
            let current_outputs: Vec<(String, ArgType)> = nodes[i]
                .outputs
                .iter()
                .map(|o| (o.name.clone(), o.ty.clone()))
                .collect();

            for output_pair in &current_outputs {
                let (output_name, output_ty) = output_pair;

                // Update all downstream nodes that use this output
                for downstream_node in &mut nodes[i + 1..] {
                    for input in &mut downstream_node.inputs {
                        if &input.name == output_name && input.ty != *output_ty {
                            types_changed = true;
                            input.ty = output_ty.clone();
                        }
                    }
                }
            }
        }

        // Step 3.5: Final sync pass to catch any cross-iteration changes
        // This handles cases where earlier nodes were updated by later nodes' outputs
        let output_types: HashMap<String, ArgType> = nodes
            .iter()
            .flat_map(|n| n.outputs.iter().map(|o| (o.name.clone(), o.ty.clone())))
            .collect();

        for node in nodes.iter_mut() {
            for input in &mut node.inputs {
                if let Some(new_type) = output_types.get(&input.name)
                    && input.ty != *new_type
                {
                    types_changed = true;
                    input.ty = new_type.clone();
                }
            }
        }

        // Step 4: Collect NEW input_preferences based on inferred types
        let mut new_preferences_found = false;

        for consumer_node in nodes.iter() {
            let processor = registry.get(&consumer_node.node_type);

            if let Ok(Some(input_prefs)) = processor.input_preferences(consumer_node, opset) {
                // For each input this consumer has preferences for
                for input in &consumer_node.inputs {
                    let requested_types = input_prefs.get(&input.name);

                    if requested_types.is_empty() {
                        continue;
                    }

                    // Find which node produces this input
                    for producer_node in nodes.iter() {
                        if let Some(output) =
                            producer_node.outputs.iter().find(|o| o.name == input.name)
                        {
                            // Check each requested preference type
                            for req_type in requested_types {
                                let pref_type_str = match req_type {
                                    ArgPreference::Scalar => "Scalar",
                                    ArgPreference::Shape => "Shape",
                                    ArgPreference::Tensor => "Tensor",
                                }
                                .to_string();

                                let key = (
                                    output.name.clone(),
                                    consumer_node.name.clone(),
                                    pref_type_str,
                                );

                                // Only add if this is a NEW preference
                                if !collected_preferences.contains(&key) {
                                    collected_preferences.insert(key.clone());
                                    new_preferences_found = true;
                                }
                            }
                            break;
                        }
                    }
                }
            }
        }

        // Step 5: Check convergence
        // Continue iterating if either types changed or new preferences were found
        if !types_changed && !new_preferences_found {
            log::debug!("Type inference converged after {} iterations", iteration);
            return Ok(());
        }
    }

    log::warn!(
        "Type inference iteration limit ({}) reached without convergence",
        max_iterations
    );
    Ok(())
}
