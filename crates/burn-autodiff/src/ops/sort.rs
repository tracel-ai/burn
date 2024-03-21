use super::{unary, Backward, Ops};
use crate::{checkpoint::base::Checkpointer, grads::Gradients};
use burn_tensor::{backend::Backend, Shape};

#[derive(Debug)]
pub(crate) struct SortDim;

impl<B: Backend, const D: usize> Backward<B, D, 1> for SortDim {
    type State = (B::IntTensorPrimitive<D>, Shape<D>);

    fn backward(
        self,
        ops: Ops<Self::State, 1>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        unary::<B, D, D, _>(ops.parents, ops.node, grads, |grad| {
            let (indices, shape) = ops.state;
            let device = B::float_device(&grad);
            let zeros = B::float_zeros(shape, &device);

            B::float_scatter(D - 1, zeros, indices, grad)
        });
    }
}
