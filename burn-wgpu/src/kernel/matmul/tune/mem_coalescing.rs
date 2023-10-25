use std::{marker::PhantomData, sync::Arc};

use burn_compute::{server::Handle, tune::AutotuneOperation};
use burn_tensor::Shape;

use crate::{
    compute::{Kernel, Server, WgpuComputeClient},
    element::WgpuElement,
    kernel::{
        build_info, into_contiguous_kernel, matmul::matmul_mem_coalescing_kernel, WORKGROUP_DEFAULT,
    },
    ops::numeric::empty_device,
    tensor::WgpuTensor,
};

pub struct MemoryCoalescingMatmulAutotuneOperation<E: WgpuElement, const D: usize> {
    into_contiguous_kernel_lhs: Arc<dyn Kernel>,
    into_contiguous_kernel_rhs: Arc<dyn Kernel>,
    memory_coalescing_kernel: Arc<dyn Kernel>,
    client: WgpuComputeClient,
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
    out: WgpuTensor<E, D>,
    _element: PhantomData<E>,
}

impl<E: WgpuElement, const D: usize> MemoryCoalescingMatmulAutotuneOperation<E, D> {
    pub fn new(
        client: WgpuComputeClient,
        lhs: WgpuTensor<E, D>,
        rhs: WgpuTensor<E, D>,
        out: WgpuTensor<E, D>,
    ) -> Self {
        Self {
            into_contiguous_kernel_lhs: into_contiguous_kernel::<E>(lhs.shape.num_elements()),
            into_contiguous_kernel_rhs: into_contiguous_kernel::<E>(rhs.shape.num_elements()),
            memory_coalescing_kernel: matmul_mem_coalescing_kernel::<E, D>(
                &lhs.shape,
                &rhs.shape,
                &out.shape,
                WORKGROUP_DEFAULT,
                WORKGROUP_DEFAULT,
            ),
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
        lhs: &WgpuTensor<E, D2>,
        rhs: &WgpuTensor<E, D2>,
        out: &WgpuTensor<E, D2>,
    ) {
        // Make lhs contiguous
        let lhs_tmp = empty_device(lhs.client.clone(), lhs.device.clone(), lhs.shape.clone());
        let lhs_into_contiguous_info = self
            .client
            .create(bytemuck::cast_slice(&build_info(&[&lhs, &lhs_tmp])));
        let lhs_into_contiguous_handles = [&lhs.handle, &lhs_tmp.handle, &lhs_into_contiguous_info];
        self.client.execute(
            self.into_contiguous_kernel_lhs.clone(),
            &lhs_into_contiguous_handles,
        );

        // Make rhs contiguous
        let rhs_tmp = empty_device(rhs.client.clone(), rhs.device.clone(), rhs.shape.clone());
        let rhs_into_contiguous_info = self
            .client
            .create(bytemuck::cast_slice(&build_info(&[&rhs, &rhs_tmp])));
        let rhs_into_contiguous_handles = [&rhs.handle, &rhs_tmp.handle, &rhs_into_contiguous_info];
        self.client.execute(
            self.into_contiguous_kernel_rhs.clone(),
            &rhs_into_contiguous_handles,
        );

        // Matmul
        let matmul_info = self
            .client
            .create(bytemuck::cast_slice(&build_info(&[&lhs, &rhs, &out])));
        let matmul_handles = [&lhs_tmp.handle, &rhs_tmp.handle, &out.handle, &matmul_info];
        self.client
            .execute(self.memory_coalescing_kernel.clone(), &matmul_handles);
    }
}

impl<E: WgpuElement, const D: usize> AutotuneOperation<Server>
    for MemoryCoalescingMatmulAutotuneOperation<E, D>
{
    fn execute(self: Box<Self>, _handles: &[&Handle<Server>]) {
        self.execution(&self.lhs, &self.rhs, &self.out)
    }

    fn execute_for_autotune(self: Box<Self>, handles: &[&Handle<Server>]) {
        self.execution(
            &WgpuTensor::new(
                self.lhs.client.clone(),
                self.lhs.device.clone(),
                reduce_shape(&self.lhs.shape),
                handles[0].clone(),
            ),
            &WgpuTensor::new(
                self.lhs.client.clone(),
                self.lhs.device.clone(),
                reduce_shape(&self.lhs.shape),
                handles[1].clone(),
            ),
            &WgpuTensor::new(
                self.lhs.client.clone(),
                self.lhs.device.clone(),
                reduce_shape(&self.lhs.shape),
                handles[2].clone(),
            ),
        )
    }

    fn autotune_handles(self: Box<Self>) -> Vec<Handle<Server>> {
        vec![
            self.client
                .create(&fill_bytes::<E, 3>(12, &reduce_shape(&self.lhs.shape))),
            self.client
                .create(&fill_bytes::<E, 3>(13, &reduce_shape(&self.rhs.shape))),
            self.client
                .empty(n_bytes::<E, 3>(&reduce_shape(&self.out.shape))),
        ]
    }

    fn clone(&self) -> Box<dyn AutotuneOperation<Server>> {
        Box::new(Self {
            into_contiguous_kernel_lhs: self.into_contiguous_kernel_lhs.clone(),
            into_contiguous_kernel_rhs: self.into_contiguous_kernel_rhs.clone(),
            memory_coalescing_kernel: self.memory_coalescing_kernel.clone(),
            client: self.client.clone(),
            lhs: self.lhs.clone(),
            rhs: self.rhs.clone(),
            out: self.out.clone(),
            _element: self._element.clone(),
        })
    }
}

fn n_bytes<E, const D: usize>(shape: &Shape<D>) -> usize {
    shape.num_elements() * core::mem::size_of::<E>()
}

fn reduce_shape<const D: usize>(shape: &Shape<D>) -> Shape<3> {
    let n_batches = 2;
    Shape::new([n_batches, shape.dims[D - 2], shape.dims[D - 1]])
}

fn fill_bytes<E: WgpuElement, const D: usize>(value: u8, shape: &Shape<D>) -> Vec<u8> {
    vec![value; n_bytes::<E, D>(shape)]
}
