use std::{marker::PhantomData, sync::Arc};

use burn_compute::{
    server::Handle,
    tune::{AutotuneKey, AutotuneOperation, AutotuneOperationSet},
};
use burn_tensor::Shape;

use crate::{
    compute::{Kernel, Server},
    element::WgpuElement,
    kernel::{
        build_info,
        matmul::{matmul_mem_coalescing_kernel, utils::shape_out},
        WORKGROUP_DEFAULT,
    },
    ops::numeric::empty_device,
    tensor::WgpuTensor,
};

/// Set of autotune operations for matmul
pub struct MatmulAutotuneOperationSet<E: WgpuElement, const D: usize> {
    key: AutotuneKey,
    lhs_tune_shape: Shape<3>,
    rhs_tune_shape: Shape<3>,
    out_tune_shape: Shape<3>,
    lhs_shape: Shape<D>,
    rhs_shape: Shape<D>,
    output_shape: Shape<D>,
    info_handle: Handle<Server>,
    _elem: PhantomData<E>,
}

impl<E: WgpuElement, const D: usize> MatmulAutotuneOperationSet<E, D> {
    /// Create a MatmulAutotuneOperationSet
    pub fn new(
        lhs_shape: Shape<D>,
        rhs_shape: Shape<D>,
        output_shape: Shape<D>,
        info_handle: Handle<Server>,
    ) -> Self {
        let m = lhs_shape.dims[D - 2];
        let k = lhs_shape.dims[D - 1];
        let n = rhs_shape.dims[D - 1];

        let batches = 3;
        let lhs_tune_shape = Shape::from([batches, m, k]);
        let rhs_tune_shape = Shape::from([batches, k, n]);
        let out_tune_shape = Shape::from([batches, m, n]);

        Self {
            key: AutotuneKey::new("matmul".to_string(), log_mkn_input_key(m, k, n)),
            lhs_tune_shape,
            rhs_tune_shape,
            out_tune_shape,
            lhs_shape,
            rhs_shape,
            output_shape,
            info_handle,
            _elem: PhantomData,
        }
    }
}

impl<E: WgpuElement, const D: usize> AutotuneOperationSet<Server>
    for MatmulAutotuneOperationSet<E, D>
{
    fn key(&self) -> AutotuneKey {
        self.key.clone()
    }

    fn autotunables(&self) -> Vec<AutotuneOperation<Server>> {
        let memory_coalescing: Arc<dyn Kernel> = matmul_mem_coalescing_kernel::<E, D>(
            &self.lhs_shape,
            &self.rhs_shape,
            &self.output_shape,
            WORKGROUP_DEFAULT,
            WORKGROUP_DEFAULT,
        );

        vec![AutotuneOperation::new(
            memory_coalescing,
            Some(vec![self.info_handle.clone()]),
        )]
    }

    fn inputs(&self) -> Vec<Vec<u8>> {
        // 12 and 13 are arbitrary numbers between 0 and 255
        vec![
            fill_bytes::<E>(12, &self.lhs_tune_shape),
            fill_bytes::<E>(13, &self.rhs_tune_shape),
            fill_bytes::<E>(0, &self.out_tune_shape),
        ]
    }

    fn fastest(&self, fastest_index: usize) -> AutotuneOperation<Server> {
        if fastest_index == 0 {
            // mem_coalescing needs into_contiguous
            // TODO: into_contiguous is not benched, must be included in operation, which
            // should be able to be made of several kernels

            let kernel = matmul_mem_coalescing_kernel::<E, D>(
                &self.lhs_shape,
                &self.rhs_shape,
                &self.output_shape,
                WORKGROUP_DEFAULT,
                WORKGROUP_DEFAULT,
            );
            AutotuneOperation::new(kernel, Some(vec![self.info_handle.clone()]))
        } else {
            panic!("Only one operation for now")
        }
    }
}

fn fill_bytes<E: WgpuElement>(value: u8, shape: &Shape<3>) -> Vec<u8> {
    let n_bytes = core::mem::size_of::<E>() * shape.num_elements();
    vec![value; n_bytes]
}

fn log_mkn_input_key(m: usize, k: usize, n: usize) -> String {
    let mut desc = String::new();

    for size in [m, k, n] {
        let exp = f32::ceil(f32::log2(size as f32)) as u32;
        desc.push_str(2_u32.pow(exp).to_string().as_str());
        desc.push(',');
    }

    desc
}

/// Executes autotune on the matmul operation
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

    let matmul_autotune = Box::new(MatmulAutotuneOperationSet::<E, D>::new(
        lhs.shape,
        rhs.shape,
        output.shape.clone(),
        info_handle.clone(),
    ));
    client.execute_autotune(matmul_autotune, &[&lhs.handle, &rhs.handle, &output.handle]);

    output
}
