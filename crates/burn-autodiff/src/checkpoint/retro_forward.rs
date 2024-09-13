use crate::graph::NodeID;

use std::{collections::HashMap, fmt::Debug, sync::Arc};

use super::state::{BackwardStates, State};

/// Definition of the forward function of a node, called during retropropagation only.
/// This is different from the normal forward function because it reads and writes from
/// the [BackwardStates] map instead of having a clear function signature.
pub trait RetroForward: Debug + Send + 'static {
    /// Applies the forward pass for retropropagation.
    fn forward(&self, states: &mut BackwardStates, out_node: NodeID);
}

#[derive(new, Debug)]
/// Links [NodeID]s to their corresponding [RetroForward]
pub(crate) struct RetroForwards {
    map: HashMap<NodeID, Arc<dyn RetroForward>>,
}

impl RetroForwards {
    /// Executes the [RetroForward] for a given [NodeID] if the node's
    /// [State] is [State::Recompute], otherwise does nothing.
    pub(crate) fn execute_retro_forward(
        &mut self,
        node_id: NodeID,
        backward_states: &mut BackwardStates,
    ) {
        if let State::Recompute { n_required: _ } = backward_states
            .get_state_ref(&node_id)
            .unwrap_or_else(|| panic!("Should find node {:?}", node_id))
        {
            // Retro forwards are always used only once because afterwards their state is computed
            let retro_forward = self.map.remove(&node_id).unwrap();
            retro_forward.forward(backward_states, node_id);
        }
    }

    #[cfg(feature = "export_tests")]
    pub(crate) fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

#[macro_export]
/// Creates a RetroForward struct for unary scalar operations
macro_rules! retro_unary_scalar {
    (
        $name:ident,
        $ops:expr
    ) => {
        #[derive(new, Debug, Clone)]
        struct $name<B: Backend, const D: usize> {
            lhs_id: NodeID,
            rhs: FloatElem<B>,
            _backend: PhantomData<B>,
        }

        impl<B: Backend, const D: usize> RetroForward for $name<B, D> {
            fn forward(&self, states: &mut BackwardStates, out_node: NodeID) {
                let lhs = states.get_state::<B::FloatTensorPrimitive<D>>(&self.lhs_id);
                let out = $ops(lhs, self.rhs);
                states.save(out_node, out)
            }
        }
    };
}

#[macro_export]
/// Creates a RetroForward struct for unary scalar operations
macro_rules! retro_unary {
    (
        $name:ident,
        $ops:expr
    ) => {
        #[derive(new, Debug, Clone)]
        struct $name<B: Backend, const D: usize> {
            input_id: NodeID,
            _backend: PhantomData<B>,
        }

        impl<B: Backend, const D: usize> RetroForward for $name<B, D> {
            fn forward(&self, states: &mut BackwardStates, out_node: NodeID) {
                let input = states.get_state::<B::FloatTensorPrimitive<D>>(&self.input_id);
                let out = $ops(input);
                states.save(out_node, out)
            }
        }
    };
}

#[macro_export]
/// Creates a RetroForward struct for binary operations
macro_rules! retro_binary {
    (
        $name:ident,
        $ops:expr
    ) => {
        #[derive(new, Debug, Clone)]
        struct $name<B: Backend, const D: usize> {
            lhs_id: NodeID,
            rhs_id: NodeID,
            _backend: PhantomData<B>,
        }

        impl<B: Backend, const D: usize> RetroForward for $name<B, D> {
            fn forward(&self, states: &mut BackwardStates, out_node: NodeID) {
                let lhs = states.get_state::<B::FloatTensorPrimitive<D>>(&self.lhs_id);
                let rhs = states.get_state::<B::FloatTensorPrimitive<D>>(&self.rhs_id);
                let out = $ops(lhs, rhs);
                states.save(out_node, out)
            }
        }
    };
}
