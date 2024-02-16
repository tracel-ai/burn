use crate::{
    grads::Gradients,
    ops::{unary, Ops, OpsKind},
    Autodiff,
};
use burn_tensor::{
    backend::Backend,
    ops::{ActivationOps, FloatTensor},
};

impl<B: Backend> ActivationOps<Autodiff<B>> for Autodiff<B> {
    fn gelu<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        todo!()
        // #[derive(Debug)]
        // struct Gelu<const D: usize>;

        // impl<const D: usize, B: Backend> Backward<B, D, 1> for Gelu<D> {
        //     type State = B::TensorPrimitive<D>;

        //     fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
        //         let input = ops.state;

        //         unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
        //             B::gelu_backward(input, grad)
        //         });
        //     }
        // }

        // match Gelu::<D>.prepare([tensor.node], [tensor.graph]).stateful() {
        //     OpsKind::Tracked(prep) => {
        //         let output = B::gelu(tensor.primitive.clone());
        //         prep.finish(tensor.primitive, output)
        //     }
        //     OpsKind::UnTracked(prep) => prep.finish(B::gelu(tensor.primitive)),
        // }
    }

    fn relu<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        todo!()
        // #[derive(Debug)]
        // struct Relu;

        // impl<B: Backend, const D: usize> Backward<B, D, 1> for Relu {
        //     type State = B::TensorPrimitive<D>;

        //     fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
        //         unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
        //             B::relu_backward(ops.state, grad)
        //         });
        //     }
        // }
        // let output = B::relu(tensor.primitive);

        // match Relu.prepare([tensor.node], [tensor.graph]).stateful() {
        //     OpsKind::Tracked(prep) => prep.finish(output.clone(), output),
        //     OpsKind::UnTracked(prep) => prep.finish(output),
        // }
    }

    fn sigmoid<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        todo!()
        // #[derive(Debug)]
        // struct Sigmoid;

        // impl<B: Backend, const D: usize> Backward<B, D, 1> for Sigmoid {
        //     type State = B::TensorPrimitive<D>;

        //     fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
        //         unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
        //             B::sigmoid_backward(ops.state, grad)
        //         });
        //     }
        // }

        // match Sigmoid.prepare([tensor.node], [tensor.graph]).stateful() {
        //     OpsKind::Tracked(prep) => {
        //         let output = B::sigmoid(tensor.primitive);
        //         prep.finish(output.clone(), output)
        //     }
        //     OpsKind::UnTracked(prep) => prep.finish(B::sigmoid(tensor.primitive)),
        // }
    }
}
