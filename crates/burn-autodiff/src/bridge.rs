use std::marker::PhantomData;

use burn_tensor::backend::{Backend, BackendBridge};

use crate::{
    checkpoint::{
        base::Checkpointer, retro_forward::RetroForward, state::BackwardStates,
        strategy::CheckpointStrategy,
    },
    grads::Gradients,
    ops::{unary_different_backend, Backward, Ops},
    Autodiff, NodeID,
};

pub struct PrecisionBridge;

impl<B: Backend, C: CheckpointStrategy> BackendBridge<Autodiff<B, C>> for PrecisionBridge {
    type Target = Autodiff<<B::FullPrecisionBridge as BackendBridge<B>>::Target, C>;

    fn into_target<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<Autodiff<B, C>, D>,
        device: burn_tensor::Device<Self::Target>,
    ) -> burn_tensor::ops::FloatTensor<Self::Target, D> {
        #[derive(Debug)]
        struct ToFullPrecision<B: Backend> {
            phantom: PhantomData<B>,
        }

        #[derive(new, Debug)]
        struct RetroToFullPrecision<B: Backend, const D: usize> {
            tensor_id: NodeID,
            _backend: PhantomData<B>,
        }

        impl<B: Backend, const D: usize> RetroForward for RetroToFullPrecision<B, D> {
            fn forward(&self, states: &mut BackwardStates, out_node: NodeID) {
                let tensor = states.get_state::<B::FloatTensorPrimitive<D>>(&self.tensor_id);
                let out = <B::FullPrecisionBridge as BackendBridge<B>>::into_target(
                    tensor,
                    Default::default(),
                );
                states.save(out_node, out)
            }
        }

        impl<B: Backend, const D: usize>
            Backward<<B::FullPrecisionBridge as BackendBridge<B>>::Target, D, 1>
            for ToFullPrecision<B>
        {
            type State = ();

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                unary_different_backend::<
                    B,
                    <B::FullPrecisionBridge as BackendBridge<B>>::Target,
                    D,
                    D,
                    _,
                >(ops.parents, ops.node, grads, |grad| {
                    <B::FullPrecisionBridge as BackendBridge<B>>::from_target(
                        grad,
                        Default::default(),
                    )
                });
            }
        }

        let ops = ToFullPrecision::<B> {
            phantom: PhantomData,
        };

        ops.prepare::<C>([tensor.node.clone()], [tensor.graph.clone()])
            .memory_bound()
            .retro_forward(RetroToFullPrecision::<B, D>::new(tensor.node.id.clone()))
            .parents([&tensor])
            .stateless(<B::FullPrecisionBridge as BackendBridge<B>>::into_target(
                tensor.primitive,
                Default::default(),
            ))
    }

    fn from_target<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<Self::Target, D>,
        device: burn_tensor::Device<Autodiff<B, C>>,
    ) -> burn_tensor::ops::FloatTensor<Autodiff<B, C>, D> {
        todo!()
    }
}
