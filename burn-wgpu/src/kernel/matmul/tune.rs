use std::{marker::PhantomData, sync::Arc};

use burn_compute::{
    server::{ComputeServer, Handle},
    tune::{AutotuneKey, AutotuneOperation, AutotuneOperationSet},
};
use burn_tensor::Shape;

use crate::{
    compute::{Kernel, Server},
    element::WgpuElement,
    kernel::{
        build_info, into_contiguous_kernel,
        matmul::{matmul_mem_coalescing_kernel, utils::shape_out},
        WORKGROUP_DEFAULT,
    },
    ops::numeric::empty_device,
    tensor::WgpuTensor,
};

#[derive(new)]
pub struct MemoryCoalescingMatmulAutotuneOperation {
    into_contiguous_kernel_lhs: Arc<dyn Kernel>,
    into_contiguous_kernel_rhs: Arc<dyn Kernel>,
    memory_coalescing_kernel: Arc<dyn Kernel>,
    matmul_info: Handle<Server>,
    contiguous_lhs_info: Handle<Server>,
    contiguous_rhs_info: Handle<Server>,
}

impl AutotuneOperation<Server> for MemoryCoalescingMatmulAutotuneOperation {
    fn execute(&self, inputs: &[&Handle<Server>], server: &mut Server) {
        let lhs_contiguous_handles = &[inputs[0], inputs[2], &self.contiguous_lhs_info];
        server.execute(
            self.into_contiguous_kernel_lhs.clone(),
            lhs_contiguous_handles,
        );

        let rhs_contiguous_handles = &[inputs[1], inputs[3], &self.contiguous_rhs_info];
        server.execute(
            self.into_contiguous_kernel_rhs.clone(),
            rhs_contiguous_handles,
        );

        let matmul_handles = &[inputs[2], inputs[3], inputs[4], &self.matmul_info];
        server.execute(self.memory_coalescing_kernel.clone(), matmul_handles);
    }
}

/// Set of autotune operations for matmul
pub struct MatmulAutotuneOperationSet<E: WgpuElement, const D: usize> {
    key: AutotuneKey,
    lhs_tune_shape: Shape<3>,
    rhs_tune_shape: Shape<3>,
    out_tune_shape: Shape<3>,
    lhs_shape: Shape<D>,
    rhs_shape: Shape<D>,
    output_shape: Shape<D>,
    matmul_info: Handle<Server>,
    contiguous_lhs_info: Handle<Server>,
    contiguous_rhs_info: Handle<Server>,
    _elem: PhantomData<E>,
}

impl<E: WgpuElement, const D: usize> MatmulAutotuneOperationSet<E, D> {
    /// Create a MatmulAutotuneOperationSet
    pub fn new(
        lhs_shape: Shape<D>,
        rhs_shape: Shape<D>,
        output_shape: Shape<D>,
        matmul_info: Handle<Server>,
        contiguous_lhs_info: Handle<Server>,
        contiguous_rhs_info: Handle<Server>,
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
            matmul_info,
            contiguous_lhs_info,
            contiguous_rhs_info,
            _elem: PhantomData,
        }
    }

    pub fn memory_coalescing<const D2: usize>(
        &self,
        lhs_shape: &Shape<D2>,
        rhs_shape: &Shape<D2>,
        out_shape: &Shape<D2>,
    ) -> Arc<MemoryCoalescingMatmulAutotuneOperation> {
        let into_contiguous_kernel_lhs: Arc<dyn Kernel> =
            into_contiguous_kernel::<E>(lhs_shape.num_elements());

        let into_contiguous_kernel_rhs: Arc<dyn Kernel> =
            into_contiguous_kernel::<E>(rhs_shape.num_elements());

        let memory_coalescing_kernel: Arc<dyn Kernel> = matmul_mem_coalescing_kernel::<E, D2>(
            &lhs_shape,
            &rhs_shape,
            &out_shape,
            WORKGROUP_DEFAULT,
            WORKGROUP_DEFAULT,
        );

        Arc::new(MemoryCoalescingMatmulAutotuneOperation::new(
            into_contiguous_kernel_lhs,
            into_contiguous_kernel_rhs,
            memory_coalescing_kernel,
            self.matmul_info.clone(),
            self.contiguous_lhs_info.clone(),
            self.contiguous_rhs_info.clone(),
        ))
    }
}

impl<E: WgpuElement, const D: usize> AutotuneOperationSet<Server>
    for MatmulAutotuneOperationSet<E, D>
{
    fn key(&self) -> AutotuneKey {
        self.key.clone()
    }

    fn autotunables(&self) -> Vec<Arc<dyn AutotuneOperation<Server>>> {
        vec![self.memory_coalescing(
            &self.lhs_tune_shape,
            &self.rhs_tune_shape,
            &self.out_tune_shape,
        )]
    }

    fn inputs(&self) -> Vec<Vec<u8>> {
        // 12 and 13 are arbitrary numbers between 0 and 255
        vec![
            fill_bytes::<E, 3>(12, &self.lhs_tune_shape),
            fill_bytes::<E, 3>(13, &self.rhs_tune_shape),
            fill_bytes::<E, 3>(0, &self.lhs_tune_shape),
            fill_bytes::<E, 3>(0, &self.rhs_tune_shape),
            fill_bytes::<E, 3>(0, &self.out_tune_shape),
        ]
    }

    fn fastest(&self, fastest_index: usize) -> Arc<dyn AutotuneOperation<Server>> {
        if fastest_index == 0 {
            self.memory_coalescing(&self.lhs_shape, &self.rhs_shape, &self.output_shape)
        } else {
            panic!("Only one operation for now")
        }
    }
}

fn fill_bytes<E: WgpuElement, const D2: usize>(value: u8, shape: &Shape<D2>) -> Vec<u8> {
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
    let matmul_info = client.create(bytemuck::cast_slice(&build_info(&[&lhs, &rhs, &output])));
    let contiguous_lhs_info = client.create(bytemuck::cast_slice(&build_info(&[&lhs, &output])));
    let contiguous_rhs_info = client.create(bytemuck::cast_slice(&build_info(&[&rhs, &output])));

    let matmul_autotune = Box::new(MatmulAutotuneOperationSet::<E, D>::new(
        lhs.shape.clone(),
        rhs.shape.clone(),
        output.shape.clone(),
        matmul_info.clone(),
        contiguous_lhs_info.clone(),
        contiguous_rhs_info.clone(),
    ));
    let lhs_2nd_handle = client.create(&fill_bytes::<E, D>(0, &lhs.shape.clone()));
    let rhs_2nd_handle = client.create(&fill_bytes::<E, D>(0, &rhs.shape.clone()));
    client.execute_autotune(
        matmul_autotune,
        &[
            &lhs.handle,
            &rhs.handle,
            &lhs_2nd_handle,
            &rhs_2nd_handle,
            &output.handle,
        ],
    );

    println!("{:?}", client.read(&lhs.handle).read());
    println!("{:?}", client.read(&rhs.handle).read());
    println!("{:?}", client.read(&lhs_2nd_handle).read());
    println!("{:?}", client.read(&rhs_2nd_handle).read());
    println!("{:?}", client.read(&output.handle).read());
    output
}
