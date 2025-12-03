//! Phase 1: Initialization
//!
//! Creates GraphState from ONNX proto structures and processes initializers into Constant nodes.

use std::{cell::RefCell, rc::Rc};

use crate::{graph_state::GraphState, ir::NodeType, protos::GraphProto};

/// Initialize GraphState with optional shared name registry
pub(crate) fn initialize_from_graph_with_registry(
    graph: &GraphProto,
    name_registry: Option<crate::graph_state::NameRegistry>,
) -> Rc<RefCell<GraphState>> {
    let state = GraphState::new_with_registry(
        &graph.input,
        &graph.output,
        &graph.initializer,
        &graph.value_info,
        name_registry,
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
