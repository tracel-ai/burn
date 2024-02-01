use std::marker::PhantomData;

use burn_tensor::{backend::Backend, Tensor};

use crate::graph::NodeID;

use super::{state::State, states::InnerStates};

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

#[derive(new)]
pub struct RetroDiv<B, const D: usize> {
    lhs: NodeID,
    rhs: NodeID,
    out: NodeID,
    _backend: PhantomData<B>,
}

impl<B: Backend, const D: usize> RetroForward for RetroDiv<B, D> {
    fn forward(&self, states: &mut InnerStates) {
        // We assume hashmap filled with parents
        let lhs: B::FloatTensorPrimitive<D> = states
            .get(&self.lhs)
            .get_state_content()
            .downcast_ref::<Tensor<B, D>>()
            .unwrap()
            .clone()
            .into_primitive();

        let rhs: B::FloatTensorPrimitive<D> = states
            .get(&self.rhs) // TODO get_mut because change num_required -=1
            .get_state_content()
            .downcast_ref::<Tensor<B, D>>()
            .unwrap()
            .clone()
            .into_primitive();

        let out: Tensor<B, D> = Tensor::<B, D>::from_primitive(B::float_div(lhs, rhs));

        states.insert(
            self.out.clone(),
            State::Computed {
                state_content: Box::new(out),
                n_required: 1, // TODO lazy's
            },
        );
    }
}
