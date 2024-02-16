use super::{unary, Ops};
use crate::grads::Gradients;
use burn_tensor::{backend::Backend, Shape};

// #[derive(Debug)]
// pub(crate) struct MaxMinDim;

// impl<B: Backend, const D: usize> Backward<B, D, 1> for MaxMinDim {
//     type State = (B::IntTensorPrimitive<D>, Shape<D>);

//     fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
//         unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
//             let (indices, shape) = ops.state;
//             let device = B::device(&grad);
//             let zeros = B::zeros(shape, &device);

//             B::scatter(D - 1, zeros, indices, grad)
//         });
//     }
// }
