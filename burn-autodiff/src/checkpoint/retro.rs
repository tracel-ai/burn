use burn_tensor::{backend::Backend, Tensor};

use crate::graph::NodeID;

use super::{state::State, base::InnerStates};

pub(crate) trait RetroForward {
    fn forward(&self, states: &mut InnerStates);
}

#[derive(new)]
pub struct RetroLeaf<B: Backend, const D: usize> {
    out: NodeID,
    tensor: Tensor<B, D>, // maybe remove that state and just have retroleaves as always computed
}

impl<B: Backend, const D: usize> RetroForward for RetroLeaf<B, D> {
    fn forward(&self, states: &mut InnerStates) {
        states.insert(
            self.out.clone(),
            State::Computed {
                state_content: Box::new(self.tensor.clone()), // must not clone tensor
                n_required: 1,                                // TODO arbitrary for now
            },
        );
    }
}
