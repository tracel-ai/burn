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
    fn gelu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct Gelu;

        retro_unary!(RetroGelu, B::gelu);

        impl<B: Backend> Backward<B, 1> for Gelu {
            type State = NodeID;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let input = checkpointer.retrieve_node_output(ops.state);

                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    B::gelu_backward(input, grad)
                });
            }
        }

        match Gelu
            .prepare::<C>([tensor.node.clone()])
            .memory_bound()
            .retro_forward(RetroGelu::<B>::new(tensor.node.id))
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

    fn relu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct Relu;

        retro_unary!(RetroRelu, B::relu);

        impl<B: Backend> Backward<B, 1> for Relu {
            type State = NodeID;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let state = checkpointer.retrieve_node_output(ops.state);
                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    B::relu_backward(state, grad)
                });
            }
        }

        match Relu
            .prepare::<C>([tensor.node.clone()])
            .memory_bound()
            .retro_forward(RetroRelu::<B>::new(tensor.node.id))
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

    fn sigmoid(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct Sigmoid;

        retro_unary!(RetroSigmoid, B::sigmoid);

        impl<B: Backend> Backward<B, 1> for Sigmoid {
            type State = NodeID;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let input = checkpointer.retrieve_node_output(ops.state);
                let output = B::sigmoid(input);
                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    B::sigmoid_backward(output, grad)
                });
            }
        }

        match Sigmoid
            .prepare::<C>([tensor.node.clone()])
            .memory_bound()
            .retro_forward(RetroSigmoid::<B>::new(tensor.node.id))
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

    fn log_sigmoid(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct LogSigmoid;

        retro_unary!(RetroLogSigmoid, B::log_sigmoid);

        impl<B: Backend> Backward<B, 1> for LogSigmoid {
            type State = NodeID;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let input = checkpointer.retrieve_node_output(ops.state);

                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    B::log_sigmoid_backward(input, grad)
                });
            }
        }

        match LogSigmoid
            .prepare::<C>([tensor.node.clone()])
            .memory_bound()
            .retro_forward(RetroLogSigmoid::<B>::new(tensor.node.id))
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
