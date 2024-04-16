use std::marker::PhantomData;

use crate::{
    checkpoint::{
        base::Checkpointer, retro_forward::RetroForward, state::BackwardStates,
        strategy::CheckpointStrategy,
    },
    grads::Gradients,
    graph::NodeID,
    ops::{unary, Backward, Ops, OpsKind},
    retro_unary, Autodiff,
};
use burn_tensor::{
    backend::Backend,
    ops::{ActivationOps, FloatTensor},
};

impl<B: Backend, C: CheckpointStrategy> ActivationOps<Autodiff<B, C>> for Autodiff<B, C> {
    fn gelu<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        #[derive(Debug)]
        struct Gelu<const D: usize>;

        retro_unary!(RetroGelu, B::gelu);

        impl<const D: usize, B: Backend> Backward<B, D, 1> for Gelu<D> {
            type State = NodeID;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let input = checkpointer.retrieve_node_output(ops.state);

                unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
                    B::gelu_backward(input, grad)
                });
            }
        }

        match Gelu::<D>
            .prepare::<C>([tensor.node.clone()])
            .memory_bound()
            .retro_forward(RetroGelu::<B, D>::new(tensor.node.id))
            .parents([&tensor])
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let state = prep.checkpoint(&tensor);
                prep.finish(state, B::gelu(tensor.primitive.clone()))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::gelu(tensor.primitive)),
        }
    }

    fn relu<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        #[derive(Debug)]
        struct Relu;

        retro_unary!(RetroRelu, B::relu);

        impl<B: Backend, const D: usize> Backward<B, D, 1> for Relu {
            type State = NodeID;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let state = checkpointer.retrieve_node_output(ops.state);
                unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
                    B::relu_backward(state, grad)
                });
            }
        }

        match Relu
            .prepare::<C>([tensor.node.clone()])
            .memory_bound()
            .retro_forward(RetroRelu::<B, D>::new(tensor.node.id))
            .parents([&tensor])
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let state = prep.checkpoint(&tensor);
                prep.finish(state, B::relu(tensor.primitive))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::relu(tensor.primitive)),
        }
    }

    fn sigmoid<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        #[derive(Debug)]
        struct Sigmoid;

        retro_unary!(RetroSigmoid, B::sigmoid);

        impl<B: Backend, const D: usize> Backward<B, D, 1> for Sigmoid {
            type State = NodeID;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let input = checkpointer.retrieve_node_output(ops.state);
                let output = B::sigmoid(input);
                unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
                    B::sigmoid_backward(output, grad)
                });
            }
        }

        match Sigmoid
            .prepare::<C>([tensor.node.clone()])
            .memory_bound()
            .retro_forward(RetroSigmoid::<B, D>::new(tensor.node.id))
            .parents([&tensor])
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let state = prep.checkpoint(&tensor);
                prep.finish(state, B::sigmoid(tensor.primitive))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::sigmoid(tensor.primitive)),
        }
    }

    fn log_sigmoid<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        #[derive(Debug)]
        struct LogSigmoid<const D: usize>;

        retro_unary!(RetroLogSigmoid, B::log_sigmoid);

        impl<const D: usize, B: Backend> Backward<B, D, 1> for LogSigmoid<D> {
            type State = NodeID;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let input = checkpointer.retrieve_node_output(ops.state);

                unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
                    B::log_sigmoid_backward(input, grad)
                });
            }
        }

        match LogSigmoid::<D>
            .prepare::<C>([tensor.node.clone()])
            .memory_bound()
            .retro_forward(RetroLogSigmoid::<B, D>::new(tensor.node.id))
            .parents([&tensor])
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let state = prep.checkpoint(&tensor);
                prep.finish(state, B::log_sigmoid(tensor.primitive.clone()))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::log_sigmoid(tensor.primitive)),
        }
    }
}
