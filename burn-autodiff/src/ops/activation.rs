use crate::{
    grads::Gradients,
    ops::{unary, Backward, Ops, OpsKind},
    tensor::ADTensor,
    ADBackendDecorator,
};
use burn_tensor::{backend::Backend, ops::ActivationOps};

impl<B: Backend> ActivationOps<ADBackendDecorator<B>> for ADBackendDecorator<B> {
    fn gelu<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Gelu<const D: usize>;

        impl<const D: usize, B: Backend> Backward<B, D, 1> for Gelu<D> {
            type State = B::TensorPrimitive<D>;

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                let input = ops.state;

                unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
                    B::gelu_backward(input, grad)
                });
            }
        }

        match Gelu::<D>.prepare([tensor.node], [tensor.graph]).statefull() {
            OpsKind::Tracked(prep) => {
                let output = B::gelu(tensor.primitive.clone());
                prep.finish(tensor.primitive, output)
            }
            OpsKind::UnTracked(prep) => prep.finish(B::gelu(tensor.primitive)),
        }
    }

    fn relu<const D: usize>(tensor: ADTensor<B, D>) -> ADTensor<B, D> {
        #[derive(Debug)]
        struct Relu;

        impl<B: Backend, const D: usize> Backward<B, D, 1> for Relu {
            type State = B::TensorPrimitive<D>;

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
                    B::relu_backward(ops.state, grad)
                });
            }
        }
        let output = B::relu(tensor.primitive);

        match Relu.prepare([tensor.node], [tensor.graph]).statefull() {
            OpsKind::Tracked(prep) => prep.finish(output.clone(), output),
            OpsKind::UnTracked(prep) => prep.finish(output),
        }
    }
}
