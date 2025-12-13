//! Phase 1: Initialization
//!
//! Creates GraphState from ONNX proto structures and processes initializers into Constant nodes.

use std::{cell::RefCell, path::Path, rc::Rc};

use crate::{graph_state::GraphState, ir::NodeType, ir::OuterScopeTypes, protos::GraphProto};

/// Initialize GraphState with optional shared name registry, outer scope types, and base path
///
/// The `outer_scope` map provides types for values that the graph references
/// from parent graphs (for subgraphs in If/Loop/Scan nodes).
///
/// The `base_path` is the directory containing the ONNX file, used for resolving
/// external tensor data paths (for models >2GB).
pub(crate) fn initialize_from_graph_with_registry_and_outer_scope(
    graph: &GraphProto,
    name_registry: Option<crate::graph_state::NameRegistry>,
    outer_scope: OuterScopeTypes,
    base_path: Option<&Path>,
) -> Rc<RefCell<GraphState>> {
    let state = GraphState::new_with_registry_and_outer_scope(
        &graph.input,
        &graph.output,
        &graph.initializer,
        &graph.value_info,
        name_registry,
        outer_scope,
        base_path,
    );

    let state_rc = Rc::new(RefCell::new(state));

    // Attach value_store to initializer constant nodes
    {
        let mut state = state_rc.borrow_mut();
        let value_store = state.build_value_store();
        for node in &mut state.processed_nodes {
            if node.node_type == NodeType::Constant {
                for arg in &mut node.inputs {
                    arg.set_value_store(value_store.clone());
                }
                for arg in &mut node.outputs {
                    arg.set_value_store(value_store.clone());
                }
            }
        }
    }

    state_rc
}
