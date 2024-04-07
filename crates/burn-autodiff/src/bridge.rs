use std::marker::PhantomData;

use burn_tensor::{
    backend::{Backend, BackendBridge},
    ops::FloatTensor,
};

use crate::{
    checkpoint::{
        base::Checkpointer, retro_forward::RetroForward, state::BackwardStates,
        strategy::CheckpointStrategy,
    },
    grads::Gradients,
    ops::{unary_different_backend, Backward, Ops},
    Autodiff, NodeID,
};

/// Enable autodiff on a [backend bridge](BackendBridge).
#[derive(Debug)]
pub struct AutodiffBridge<Bridge> {
    _p: PhantomData<Bridge>,
}

impl<B, C, Bridge> BackendBridge<Autodiff<B, C>> for AutodiffBridge<Bridge>
where
    B: Backend,
    C: CheckpointStrategy,
    Bridge: BackendBridge<B> + 'static,
{
    type Target = Autodiff<Bridge::Target, C>;

    fn into_target<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<Autodiff<B, C>, D>,
        _device: Option<burn_tensor::Device<Self::Target>>,
    ) -> burn_tensor::ops::FloatTensor<Self::Target, D> {
        #[derive(Debug)]
        struct IntoTarget<B: Backend, Bridge: BackendBridge<B>> {
            _backend: PhantomData<B>,
            _bridge: PhantomData<Bridge>,
        }

        #[derive(new, Debug, Clone)]
        struct RetroIntoTarget<B: Backend, Bridge: BackendBridge<B>, const D: usize> {
            tensor_id: NodeID,
            _backend: PhantomData<B>,
            _bridge: PhantomData<Bridge>,
        }

        impl<B, Bridge, const D: usize> RetroForward for RetroIntoTarget<B, Bridge, D>
        where
            B: Backend,
            Bridge: BackendBridge<B> + 'static,
        {
            fn forward(&self, states: &mut BackwardStates, out_node: NodeID) {
                let tensor: FloatTensor<B, D> = states.get_state(&self.tensor_id);
                let out = Bridge::into_target(tensor, Default::default());
                states.save(out_node, out)
            }
        }

        impl<B, Bridge, const D: usize> Backward<Bridge::Target, D, 1> for IntoTarget<B, Bridge>
        where
            B: Backend,
            Bridge: BackendBridge<B> + 'static,
        {
            type State = ();

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                unary_different_backend::<B, Bridge::Target, D, D, _>(
                    ops.parents,
                    ops.node,
                    grads,
                    |grad| Bridge::from_target(grad, None),
                );
            }
        }

        IntoTarget::<B, Bridge> {
            _backend: PhantomData,
            _bridge: PhantomData,
        }
        .prepare::<C>([tensor.node.clone()])
        .memory_bound()
        .retro_forward(RetroIntoTarget::<B, Bridge, D>::new(tensor.node.id))
        .parents([&tensor])
        .stateless(Bridge::into_target(tensor.primitive, None))
    }

    fn from_target<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<Self::Target, D>,
        _device: Option<burn_tensor::Device<Autodiff<B, C>>>,
    ) -> burn_tensor::ops::FloatTensor<Autodiff<B, C>, D> {
        #[derive(Debug)]
        struct FromTarget<B: Backend, Bridge: BackendBridge<B>> {
            _backend: PhantomData<B>,
            _bridge: PhantomData<Bridge>,
        }

        #[derive(new, Debug, Clone)]
        struct RetroFromTarget<B: Backend, Bridge: BackendBridge<B>, const D: usize> {
            tensor_id: NodeID,
            _backend: PhantomData<B>,
            _bridge: PhantomData<Bridge>,
        }

        impl<B, Bridge, const D: usize> RetroForward for RetroFromTarget<B, Bridge, D>
        where
            B: Backend,
            Bridge: BackendBridge<B> + 'static,
        {
            fn forward(&self, states: &mut BackwardStates, out_node: NodeID) {
                let tensor: FloatTensor<Bridge::Target, D> = states.get_state(&self.tensor_id);
                let out = Bridge::from_target(tensor, None);
                states.save(out_node, out)
            }
        }

        impl<B, Bridge, const D: usize> Backward<B, D, 1> for FromTarget<B, Bridge>
        where
            B: Backend,
            Bridge: BackendBridge<B> + 'static,
        {
            type State = ();

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                unary_different_backend::<Bridge::Target, B, D, D, _>(
                    ops.parents,
                    ops.node,
                    grads,
                    |grad| Bridge::into_target(grad, None),
                );
            }
        }

        FromTarget::<B, Bridge> {
            _backend: PhantomData,
            _bridge: PhantomData,
        }
        .prepare::<C>([tensor.node.clone()])
        .memory_bound()
        .retro_forward(RetroFromTarget::<B, Bridge, D>::new(tensor.node.id))
        .parents([&tensor])
        .stateless(Bridge::from_target(tensor.primitive, None))
    }
}
