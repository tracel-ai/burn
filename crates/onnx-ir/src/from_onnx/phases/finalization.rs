//! Phase 5: Finalization
//!
//! Removes unreferenced constant nodes and builds the final OnnxGraph.

use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    rc::Rc,
};

use crate::{
    from_onnx::graph_state::GraphState,
    ir::{Argument, Node, NodeType, OnnxGraph},
};

/// Finalize the graph by removing unused constants and building OnnxGraph
pub(crate) fn finalize(
    nodes: &mut Vec<Node>,
    inputs: Vec<Argument>,
    outputs: &mut Vec<Argument>,
    state_rc: Rc<RefCell<GraphState>>,
) -> OnnxGraph {
    remove_unreferenced_constants(nodes, outputs);

    OnnxGraph {
        nodes: std::mem::take(nodes),
        inputs,
        outputs: std::mem::take(outputs),
        _graph_data: Some(state_rc),
    }
}

/// Remove constant nodes that have zero runtime references
fn remove_unreferenced_constants(nodes: &mut Vec<Node>, outputs: &[Argument]) {
    log::debug!("Counting references to constant nodes");

    // Build map of constant output names to node indices
    let mut constant_output_to_idx: HashMap<String, usize> = HashMap::new();
    for (idx, node) in nodes.iter().enumerate() {
        if node.node_type == NodeType::Constant {
            for output in &node.outputs {
                constant_output_to_idx.insert(output.name.clone(), idx);
            }
        }
    }

    // Count references (only Constant/Dynamic, not Static)
    let mut constant_references: HashMap<String, usize> = HashMap::new();

    for node in nodes.iter() {
        for input in &node.inputs {
            if (input.is_constant() || input.is_dynamic())
                && constant_output_to_idx.contains_key(&input.name)
            {
                *constant_references.entry(input.name.clone()).or_insert(0) += 1;
            }
        }
    }

    for output in outputs {
        if (output.is_constant() || output.is_dynamic())
            && constant_output_to_idx.contains_key(&output.name)
        {
            *constant_references.entry(output.name.clone()).or_insert(0) += 1;
        }
    }

    // Mark constants with zero references for removal
    let mut constants_to_remove = HashSet::new();
    for (output_name, &node_idx) in &constant_output_to_idx {
        let ref_count = constant_references.get(output_name).unwrap_or(&0);
        if *ref_count == 0 {
            constants_to_remove.insert(node_idx);
            log::debug!(
                "Marking constant at index {} (output: {}) for removal (no references)",
                node_idx,
                output_name
            );
        }
    }

    // Filter out unreferenced constants
    log::debug!(
        "Filtering nodes: total={}, removing={}",
        nodes.len(),
        constants_to_remove.len()
    );

    let mut i = 0;
    nodes.retain(|node| {
        let keep = !constants_to_remove.contains(&i);
        if !keep {
            log::debug!("Filtering out node at index {}: {}", i, node.name);
        }
        i += 1;
        keep
    });

    log::debug!("After filtering: {} nodes remain", nodes.len());
}
