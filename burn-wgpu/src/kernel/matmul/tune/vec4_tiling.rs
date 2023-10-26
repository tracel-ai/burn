use std::marker::PhantomData;

use burn_compute::tune::AutotuneOperation;
use burn_tensor::Element;

use crate::{
    element::WgpuElement, kernel::matmul::vec4_primitive::matmul_tiling_2d_vec4_primitive_default,
    tensor::WgpuTensor,
};

#[derive(new)]
pub struct Vec4TilingMatmulAutotuneOperation<E: WgpuElement, const D: usize> {
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
    out: WgpuTensor<E, D>,
    _element: PhantomData<E>,
}

impl<E: WgpuElement + Element, const D: usize> AutotuneOperation
    for Vec4TilingMatmulAutotuneOperation<E, D>
{
    fn execute(self: Box<Self>) {
        matmul_tiling_2d_vec4_primitive_default(self.lhs, self.rhs, self.out);
    }

    fn clone(&self) -> Box<dyn AutotuneOperation> {
        Box::new(Self {
            lhs: self.lhs.clone(),
            rhs: self.rhs.clone(),
            out: self.out.clone(),
            _element: self._element.clone(),
        })
    }
}
