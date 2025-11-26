use super::{Backward, Ops, unary};
use crate::{checkpoint::base::Checkpointer, grads::Gradients};
use burn_tensor::{Shape, TensorMetadata, backend::Backend};

#[derive(Debug)]
pub(crate) struct SortDim;

impl<B: Backend> Backward<B, 1> for SortDim {
    type State = (B::IntTensorPrimitive, Shape, usize);

    fn backward(
        self,
        ops: Ops<Self::State, 1>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        unary::<B, _>(ops.parents, ops.node, grads, |grad| {
            let (indices, shape, dim) = ops.state;
            let device = B::float_device(&grad);
            let dtype = grad.dtype();
            let zeros = B::float_zeros(shape, &device, dtype.into());

            B::float_scatter_add(dim, zeros, indices, grad)
        });
    }
}
