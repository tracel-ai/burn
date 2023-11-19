use burn_tensor::ops::{ActivationOps, FloatTensor};

use crate::{
    element::{FloatElement, IntElement},
    kernel::{unary_default, unary_inplace_default},
    unary, unary_inplace, GraphicsApi, Wgpu,
};

impl<G, F, I> ActivationOps<Wgpu<G, F, I>> for Wgpu<G, F, I>
where
    G: GraphicsApi + 'static,
    F: FloatElement,
    I: IntElement,
{
    fn relu<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary!(Relu, body "output[id] = max(input[id], 0.0);");
        unary_inplace!(ReluInplace, body "input[id] = max(input[id], 0.0);");

        if tensor.can_mut() {
            return unary_inplace_default::<ReluInplace, F, D>(tensor);
        }

        unary_default::<Relu, F, D>(tensor)
    }
}
