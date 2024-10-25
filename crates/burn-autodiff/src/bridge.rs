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

    fn into_target(
        tensor: burn_tensor::ops::FloatTensor<Autodiff<B, C>>,
        _device: Option<burn_tensor::Device<Self::Target>>,
    ) -> burn_tensor::ops::FloatTensor<Self::Target> {
        #[derive(Debug)]
        struct IntoTarget<B: Backend, Bridge: BackendBridge<B>> {
            _backend: PhantomData<B>,
            _bridge: PhantomData<Bridge>,
        }

        #[derive(new, Debug, Clone)]
        struct RetroIntoTarget<B: Backend, Bridge: BackendBridge<B>> {
            tensor_id: NodeID,
            _backend: PhantomData<B>,
            _bridge: PhantomData<Bridge>,
        }

        impl<B, Bridge> RetroForward for RetroIntoTarget<B, Bridge>
        where
            B: Backend,
            Bridge: BackendBridge<B> + 'static,
        {
            fn forward(&self, states: &mut BackwardStates, out_node: NodeID) {
                let tensor: FloatTensor<B> = states.get_state(&self.tensor_id);
                let out = Bridge::into_target(tensor, Default::default());
                states.save(out_node, out)
            }
        }

        impl<B, Bridge> Backward<Bridge::Target, 1> for IntoTarget<B, Bridge>
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
                unary_different_backend::<B, Bridge::Target, _>(
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
        .retro_forward(RetroIntoTarget::<B, Bridge>::new(tensor.node.id))
        .parents([&tensor])
        .stateless(Bridge::into_target(tensor.primitive, None))
    }

    fn from_target(
        tensor: burn_tensor::ops::FloatTensor<Self::Target>,
        _device: Option<burn_tensor::Device<Autodiff<B, C>>>,
    ) -> burn_tensor::ops::FloatTensor<Autodiff<B, C>> {
        #[derive(Debug)]
        struct FromTarget<B: Backend, Bridge: BackendBridge<B>> {
            _backend: PhantomData<B>,
            _bridge: PhantomData<Bridge>,
        }

        #[derive(new, Debug, Clone)]
        struct RetroFromTarget<B: Backend, Bridge: BackendBridge<B>> {
            tensor_id: NodeID,
            _backend: PhantomData<B>,
            _bridge: PhantomData<Bridge>,
        }

        impl<B, Bridge> RetroForward for RetroFromTarget<B, Bridge>
        where
            B: Backend,
            Bridge: BackendBridge<B> + 'static,
        {
            fn forward(&self, states: &mut BackwardStates, out_node: NodeID) {
                let tensor: FloatTensor<Bridge::Target> = states.get_state(&self.tensor_id);
                let out = Bridge::from_target(tensor, None);
                states.save(out_node, out)
            }
        }

        impl<B, Bridge> Backward<B, 1> for FromTarget<B, Bridge>
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
                unary_different_backend::<Bridge::Target, B, _>(
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
        .retro_forward(RetroFromTarget::<B, Bridge>::new(tensor.node.id))
        .parents([&tensor])
        .stateless(Bridge::from_target(tensor.primitive, None))
    }
}
