use std::{marker::PhantomData, };

use burn_compute::{server::Handle, tune::AutotuneOperation};

use crate::{
    compute::{Server, WgpuComputeClient},
    element::WgpuElement,
    kernel::{
        matmul::{
            fill_bytes, matmul_mem_coalescing_default,
        },
    },
    tensor::WgpuTensor,
};

use super::reduce_shape;

pub struct MemoryCoalescingMatmulAutotuneOperation<E: WgpuElement, const D: usize> {
    client: WgpuComputeClient,
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
    out: WgpuTensor<E, D>,
    _element: PhantomData<E>,
}

impl<E: WgpuElement, const D: usize> MemoryCoalescingMatmulAutotuneOperation<E, D> {
    pub fn new(client: WgpuComputeClient, lhs: WgpuTensor<E, D>, rhs: WgpuTensor<E, D>, out: WgpuTensor<E, D>) -> Self {
        Self {
            client,
            lhs,
            rhs,
            out,
            _element: PhantomData,
        }
    }
}

impl<E: WgpuElement, const D: usize> MemoryCoalescingMatmulAutotuneOperation<E, D> {
    fn execution<const D2: usize>(
        &self,
        lhs: WgpuTensor<E, D2>,
        rhs: WgpuTensor<E, D2>,
    ) -> WgpuTensor<E, D2> {
        matmul_mem_coalescing_default(lhs, rhs)
    }
}

impl<E: WgpuElement, const D: usize> AutotuneOperation<Server>
    for MemoryCoalescingMatmulAutotuneOperation<E, D>
{
    fn execute(self: Box<Self>) {
        let out = self.execution(self.lhs.clone(), self.rhs.clone())
        self.client.
    }

    fn execute_for_autotune(self: Box<Self>, handles: &[&Handle<Server>]) {
        self.execution(
            WgpuTensor::new(
                self.lhs.client.clone(),
                self.lhs.device.clone(),
                reduce_shape(&self.lhs.shape),
                handles[0].clone(),
            ),
            WgpuTensor::new(
                self.lhs.client.clone(),
                self.lhs.device.clone(),
                reduce_shape(&self.lhs.shape),
                handles[1].clone(),
            ),
        )
    }

    fn clone(&self) -> Box<dyn AutotuneOperation<Server>> {
        Box::new(Self {
            client: self.client.clone(),
            lhs: self.lhs.clone(),
            rhs: self.rhs.clone(),
            _element: self._element.clone(),
        })
    }
}
