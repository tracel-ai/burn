use std::marker::PhantomData;

use burn_compute::tune::{AutotuneKey, AutotuneOperation, AutotuneOperationSet};
use burn_tensor::Shape;

use crate::{
    compute::{Server, WgpuComputeClient},
    element::WgpuElement,
    kernel::matmul::{utils::shape_out, MemoryCoalescingMatmulAutotuneOperation},
    ops::numeric::empty_device,
    tensor::WgpuTensor,
};

pub struct MatmulAutotuneOperationSet<E: WgpuElement, const D: usize> {
    client: WgpuComputeClient,
    key: AutotuneKey,
    lhs_shape: Shape<D>,
    rhs_shape: Shape<D>,
    out_shape: Shape<D>,
    _element: PhantomData<E>,
}
impl<E: WgpuElement, const D: usize> MatmulAutotuneOperationSet<E, D> {
    fn new(
        client: WgpuComputeClient,
        lhs_shape: Shape<D>,
        rhs_shape: Shape<D>,
        out_shape: Shape<D>,
    ) -> Self {
        let m = lhs_shape.dims[D - 2];
        let k = lhs_shape.dims[D - 1];
        let n = rhs_shape.dims[D - 1];
        Self {
            key: AutotuneKey::new("matmul".to_string(), log_mkn_input_key(m, k, n)),
            client,
            lhs_shape,
            rhs_shape,
            out_shape,
            _element: PhantomData,
        }
    }
}

impl<E: WgpuElement, const D: usize> AutotuneOperationSet<Server>
    for MatmulAutotuneOperationSet<E, D>
{
    fn key(&self) -> AutotuneKey {
        self.key.clone()
    }

    fn autotunables(&self) -> Vec<Box<dyn AutotuneOperation<Server>>> {
        vec![Box::new(
            MemoryCoalescingMatmulAutotuneOperation::<E, D>::new(
                self.client.clone(),
                self.lhs_shape.clone(),
                self.rhs_shape.clone(),
                self.out_shape.clone(),
            ),
        )]
    }

    fn fastest(&self, fastest_index: usize) -> Box<dyn AutotuneOperation<Server>> {
        // TODO don't recreate all autotunables
        self.autotunables()[fastest_index].clone()
    }
}

pub fn matmul_autotune<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    let client = lhs.client.clone();
    let output_shape = shape_out(&lhs, &rhs);

    let output = empty_device(client.clone(), lhs.device.clone(), output_shape.clone());

    let operation_set = Box::new(MatmulAutotuneOperationSet::<E, D>::new(
        client.clone(),
        lhs.shape,
        rhs.shape,
        output_shape,
    ));

    let handles = [&lhs.handle, &rhs.handle, &output.handle];
    client.execute_autotune(operation_set, &handles);

    output
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
