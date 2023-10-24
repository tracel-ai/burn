use std::{marker::PhantomData, sync::Arc};

use burn_compute::{
    memory_management::{MemoryManagement, SimpleMemoryManagement},
    server::Handle,
    tune::{AutotuneKey, AutotuneOperationSet, AutotuneOperation},
};
use burn_tensor::Shape;

use crate::{
    compute::{Kernel, WgpuServer, WgpuStorage},
    element::WgpuElement,
    kernel::{
        build_info,
        matmul::{matmul_mem_coalescing_kernel, utils::shape_out},
        WORKGROUP_DEFAULT,
    },
    ops::numeric::empty_device,
    tensor::WgpuTensor,
};

pub struct MatmulAutotuneOperation<
    MM: MemoryManagement<WgpuStorage>,
    E: WgpuElement,
    const D: usize,
> {
    key: AutotuneKey,
    lhs_shape: Shape<D>,
    rhs_shape: Shape<D>,
    output_shape: Shape<D>,
    info_handle: Handle<WgpuServer<MM>>,
    _elem: PhantomData<E>,
}

impl<MM: MemoryManagement<WgpuStorage>, E: WgpuElement, const D: usize>
    MatmulAutotuneOperation<MM, E, D>
{
    pub fn new(
        lhs_shape: Shape<D>,
        rhs_shape: Shape<D>,
        output_shape: Shape<D>,
        info_handle: Handle<WgpuServer<MM>>,
    ) -> Self {
        Self {
            key: AutotuneKey::new(
                "matmul".to_string(),
                log_mkn_input_key(&lhs_shape, &rhs_shape),
            ),
            lhs_shape,
            rhs_shape,
            output_shape,
            info_handle,
            _elem: PhantomData,
        }
    }
}

impl<MM: MemoryManagement<WgpuStorage>, E: WgpuElement, const D: usize>
    AutotuneOperationSet<WgpuServer<MM>> for MatmulAutotuneOperation<MM, E, D>
{
    fn key(&self) -> AutotuneKey {
        self.key.clone()
    }

    fn autotunables(&self) -> Vec<AutotuneOperation<WgpuServer<MM>>> {
        // let memory_coalescing: S::Kernel = matmul_mem_coalescing_kernel(
        let memory_coalescing: Arc<dyn Kernel> = matmul_mem_coalescing_kernel::<E, D>(
            &self.lhs_shape,
            &self.rhs_shape,
            &self.output_shape,
            WORKGROUP_DEFAULT,
            WORKGROUP_DEFAULT,
        );

        let x = memory_coalescing;
        let y = Some(vec![self.info_handle.clone()]);
        let o = AutotuneOperation::new(x, y);

        vec![o]
    }

    fn inputs(&self) -> Vec<Vec<u8>> {
        // 12 and 13 are arbitrary numbers between 0 and 255
        // TODO use small hard coded batch size
        vec![
            fill_bytes::<E, D>(12, &self.lhs_shape),
            fill_bytes::<E, D>(13, &self.rhs_shape),
            fill_bytes::<E, D>(0, &self.output_shape),
        ]
    }

    fn fastest(&self, fastest_index: usize) -> AutotuneOperation<WgpuServer<MM>> {
        // If mem_coaslescong chosen, call into contiguous, use right batch size
        self.autotunables()[fastest_index].clone()
    }
}

fn fill_bytes<E: WgpuElement, const D: usize>(value: u8, shape: &Shape<D>) -> Vec<u8> {
    let n_bytes = core::mem::size_of::<E>() * shape.num_elements();
    vec![value; n_bytes]
}

fn log_mkn_input_key<const D: usize>(lhs_shape: &Shape<D>, rhs_shape: &Shape<D>) -> String {
    let mut desc = String::new();
    let m = lhs_shape.dims[D - 2];
    let k = lhs_shape.dims[D - 1];
    let n = rhs_shape.dims[D - 1];
    let mkn = [m, k, n];

    for size in mkn {
        let exp = f32::ceil(f32::log2(size as f32)) as u32;
        desc.push_str(2_u32.pow(exp).to_string().as_str());
        desc.push(',');
    }

    desc
}

pub fn matmul_autotune<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    let output = empty_device(
        lhs.client.clone(),
        lhs.device.clone(),
        shape_out(&lhs, &rhs),
    );

    let client = lhs.client.clone();
    let info = build_info(&[&lhs, &rhs, &output]);
    let info_handle = client.create(bytemuck::cast_slice(&info));

    let matmul_autotune = Box::new(MatmulAutotuneOperation::<
        SimpleMemoryManagement<WgpuStorage>, // TODO remove
        E,
        D,
    >::new(
        lhs.shape,
        rhs.shape,
        output.shape.clone(),
        info_handle.clone(),
    ));
    client.execute_autotune(matmul_autotune, &[&lhs.handle, &rhs.handle, &output.handle]);

    output
}
