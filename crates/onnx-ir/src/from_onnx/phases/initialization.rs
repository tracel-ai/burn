//! Phase 1: Initialization
//!
//! Creates GraphState from ONNX proto structures and processes initializers into Constant nodes.

use std::{cell::RefCell, rc::Rc};

use crate::{ir::NodeType, protos::ModelProto};

use super::super::graph_state::GraphState;

/// Initialize graph state from ONNX model
///
/// Creates GraphState, processes initializers into Constant nodes,
/// and sets up value_store references.
pub(crate) fn initialize(model: &ModelProto) -> Rc<RefCell<GraphState>> {
    let state = GraphState::new(
        &model.graph.input,
        &model.graph.output,
        &model.graph.initializer,
    );

    let state_rc = Rc::new(RefCell::new(state));

    // Attach value_store to initializer constant nodes
    {
        let mut state = state_rc.borrow_mut();
        for node in &mut state.processed_nodes {
            if node.node_type == NodeType::Constant {
                for arg in &mut node.inputs {
                    arg.value_store = Some(state_rc.clone());
                }
                for arg in &mut node.outputs {
                    arg.value_store = Some(state_rc.clone());
                }
            }
        }
    }

    log::debug!(
        "Initialized state with {} initializers",
        model.graph.initializer.len()
    );
    state_rc
}
