use std::marker::PhantomData;

use burn_compute::tune::AutotuneOperation;

use crate::{
    compute::Server, element::WgpuElement, kernel::matmul::matmul_mem_coalescing_default,
    tensor::WgpuTensor,
};

#[derive(new)]
pub struct MemoryCoalescingMatmulAutotuneOperation<E: WgpuElement, const D: usize> {
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
    out: WgpuTensor<E, D>,
    _element: PhantomData<E>,
}

impl<E: WgpuElement, const D: usize> AutotuneOperation<Server>
    for MemoryCoalescingMatmulAutotuneOperation<E, D>
{
    fn execute(self: Box<Self>) {
        matmul_mem_coalescing_default(self.lhs, self.rhs, self.out);
    }

    fn clone(&self) -> Box<dyn AutotuneOperation<Server>> {
        Box::new(Self {
            lhs: self.lhs.clone(),
            rhs: self.rhs.clone(),
            out: self.out.clone(),
            _element: self._element.clone(),
        })
    }
}
