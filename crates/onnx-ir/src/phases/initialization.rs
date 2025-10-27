//! Phase 1: Initialization
//!
//! Creates GraphState from ONNX proto structures and processes initializers into Constant nodes.

use std::{cell::RefCell, rc::Rc};

use crate::{
    graph_state::GraphState,
    ir::NodeType,
    protos::{GraphProto, ModelProto},
};

/// Initialize GraphState, process initializers, attach value_store refs
pub(crate) fn initialize(model: &ModelProto) -> Rc<RefCell<GraphState>> {
    initialize_from_graph(&model.graph)
}

/// Initialize GraphState from GraphProto (for subgraphs)
pub(crate) fn initialize_from_graph(graph: &GraphProto) -> Rc<RefCell<GraphState>> {
    let state = GraphState::new(
        &graph.input,
        &graph.output,
        &graph.initializer,
        &graph.value_info,
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

    state_rc
}
