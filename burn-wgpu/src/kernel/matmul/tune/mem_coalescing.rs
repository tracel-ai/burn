use std::{marker::PhantomData, sync::Arc};

use burn_compute::{server::Handle, tune::AutotuneOperation};
use burn_tensor::Shape;

use crate::{
    compute::{Kernel, Server, WgpuComputeClient},
    element::WgpuElement,
    kernel::{
        build_info, into_contiguous_kernel, matmul::matmul_mem_coalescing_kernel, WORKGROUP_DEFAULT,
    },
};

pub struct MemoryCoalescingMatmulAutotuneOperation<E, const D: usize> {
    into_contiguous_kernel_lhs: Arc<dyn Kernel>,
    into_contiguous_kernel_rhs: Arc<dyn Kernel>,
    memory_coalescing_kernel: Arc<dyn Kernel>,
    client: WgpuComputeClient,
    lhs_shape: Shape<D>,
    rhs_shape: Shape<D>,
    out_shape: Shape<D>,
    _element: PhantomData<E>,
}

impl<E: WgpuElement, const D: usize> MemoryCoalescingMatmulAutotuneOperation<E, D> {
    pub fn new(
        client: WgpuComputeClient,
        lhs_shape: Shape<D>,
        rhs_shape: Shape<D>,
        out_shape: Shape<D>,
    ) -> Self {
        Self {
            into_contiguous_kernel_lhs: into_contiguous_kernel::<E>(lhs_shape.num_elements()),
            into_contiguous_kernel_rhs: into_contiguous_kernel::<E>(rhs_shape.num_elements()),
            memory_coalescing_kernel: matmul_mem_coalescing_kernel::<E, D>(
                &lhs_shape,
                &rhs_shape,
                &out_shape,
                WORKGROUP_DEFAULT,
                WORKGROUP_DEFAULT,
            ),
            client,
            lhs_shape,
            rhs_shape,
            out_shape,
            _element: PhantomData,
        }
    }
}

impl<E: WgpuElement, const D: usize> MemoryCoalescingMatmulAutotuneOperation<E, D> {
    fn execution<const D2: usize>(
        &self,
        handles: &[&Handle<Server>],
        lhs_shape: &Shape<D2>,
        rhs_shape: &Shape<D2>,
    ) {
        // Make lhs contiguous
        let lhs_tmp = self.client.empty(n_bytes::<E, D2>(lhs_shape));
        let lhs_into_contiguous_info = self
            .client
            .create(bytemuck::cast_slice(&build_info(&[&lhs, &output])));
        let lhs_into_contiguous_handles = [handles[0], &lhs_tmp];
        self.client.execute(
            self.into_contiguous_kernel_lhs.clone(),
            &lhs_into_contiguous_handles,
        );

        // Make rhs contiguous
        let rhs_tmp = self.client.empty(n_bytes::<E, D2>(rhs_shape));
        let rhs_into_contiguous_handles = [handles[1], &rhs_tmp];
        self.client.execute(
            self.into_contiguous_kernel_rhs.clone(),
            &rhs_into_contiguous_handles,
        );

        // Matmul
        let matmul_handles = [&lhs_tmp, &rhs_tmp, handles[2]];
        self.client
            .execute(self.memory_coalescing_kernel.clone(), &matmul_handles);
    }
}

impl<E: WgpuElement, const D: usize> AutotuneOperation<Server>
    for MemoryCoalescingMatmulAutotuneOperation<E, D>
{
    fn execute(self: Box<Self>, handles: &[&Handle<Server>]) {
        self.execution(handles, &self.lhs_shape, &self.rhs_shape)
    }

    fn execute_for_autotune(self: Box<Self>, handles: &[&Handle<Server>]) {
        self.execution(
            handles,
            &reduce_shape(&self.lhs_shape),
            &reduce_shape(&self.rhs_shape),
        )
    }

    fn autotune_handles(self: Box<Self>) -> Vec<Handle<Server>> {
        vec![
            self.client
                .create(&fill_bytes::<E, 3>(12, &reduce_shape(&self.lhs_shape))),
            self.client
                .create(&fill_bytes::<E, 3>(13, &reduce_shape(&self.rhs_shape))),
            self.client
                .empty(n_bytes::<E, 3>(&reduce_shape(&self.out_shape))),
        ]
    }

    fn clone(&self) -> Box<dyn AutotuneOperation<Server>> {
        Box::new(Self {
            into_contiguous_kernel_lhs: self.into_contiguous_kernel_lhs.clone(),
            into_contiguous_kernel_rhs: self.into_contiguous_kernel_rhs.clone(),
            memory_coalescing_kernel: self.memory_coalescing_kernel.clone(),
            client: self.client.clone(),
            lhs_shape: self.lhs_shape.clone(),
            rhs_shape: self.rhs_shape.clone(),
            out_shape: self.out_shape.clone(),
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
