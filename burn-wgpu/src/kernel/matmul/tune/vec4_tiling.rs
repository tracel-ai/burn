use std::{marker::PhantomData, sync::Arc};

use burn_compute::{server::Handle, tune::AutotuneOperation};

use crate::{
    compute::{Kernel, Server, WgpuComputeClient},
    element::WgpuElement,
    kernel::{
        build_info,
        matmul::{fill_bytes, n_bytes},
    },
    tensor::WgpuTensor,
};

use super::reduce_shape;

pub struct Vec4TilingMatmulAutotuneOperation<E: WgpuElement, const D: usize> {
    vec4_tiling_matmul_kernel: Arc<dyn Kernel>,
    client: WgpuComputeClient,
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
    out: WgpuTensor<E, D>,
    _element: PhantomData<E>,
}

impl<E: WgpuElement, const D: usize> Vec4TilingMatmulAutotuneOperation<E, D> {
    pub fn new(
        client: WgpuComputeClient,
        lhs: WgpuTensor<E, D>,
        rhs: WgpuTensor<E, D>,
        out: WgpuTensor<E, D>,
    ) -> Self {
        Self {
            vec4_tiling_matmul_kernel: vec4_tiling_matmul_kernel::<E, D>(),
            client,
            lhs,
            rhs,
            out,
            _element: PhantomData,
        }
    }
}

impl<E: WgpuElement, const D: usize> Vec4TilingMatmulAutotuneOperation<E, D> {
    fn execution<const D2: usize>(
        &self,
        lhs: &WgpuTensor<E, D2>,
        rhs: &WgpuTensor<E, D2>,
        out: &WgpuTensor<E, D2>,
    ) {
        // Matmul
        let matmul_info = self
            .client
            .create(bytemuck::cast_slice(&build_info(&[&lhs, &rhs, &out])));
        let matmul_handles = [&lhs.handle, &rhs.handle, &out.handle, &matmul_info];
        self.client
            .execute(self.vec4_tiling_matmul_kernel.clone(), &matmul_handles);
    }
}

impl<E: WgpuElement, const D: usize> AutotuneOperation<Server>
    for Vec4TilingMatmulAutotuneOperation<E, D>
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
            vec4_tiling_matmul_kernel: self.vec4_tiling_matmul_kernel.clone(),
            client: self.client.clone(),
            lhs: self.lhs.clone(),
            rhs: self.rhs.clone(),
            out: self.out.clone(),
            _element: self._element.clone(),
        })
    }
}
