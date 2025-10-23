//! Phase 1: Initialization
//!
//! Creates GraphState from ONNX proto structures and processes initializers into Constant nodes.

use std::{cell::RefCell, rc::Rc};

use crate::{graph_state::GraphState, ir::NodeType, protos::ModelProto};

/// Initialize GraphState, process initializers, attach value_store refs
pub(crate) fn initialize(model: &ModelProto) -> Rc<RefCell<GraphState>> {
    let state = GraphState::new(
        &model.graph.input,
        &model.graph.output,
        &model.graph.initializer,
        &model.graph.value_info,
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
        "Initialized state with {} initializers and {} value_info entries",
        model.graph.initializer.len(),
        model.graph.value_info.len()
    );
    state_rc
}
