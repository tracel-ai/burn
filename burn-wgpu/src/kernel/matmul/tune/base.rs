use std::marker::PhantomData;

use burn_compute::tune::{AutotuneKey, AutotuneOperation, AutotuneOperationSet};
use burn_tensor::Element;

use crate::{
    element::WgpuElement,
    kernel::matmul::{
        autotune_tensors, utils::init_matrix_output, MemoryCoalescingMatmulAutotuneOperation,
        Vec4TilingMatmulAutotuneOperation,
    },
    tensor::WgpuTensor,
};

pub struct MatmulAutotuneOperationSet<E: WgpuElement, const D: usize> {
    key: AutotuneKey,
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
    out: WgpuTensor<E, D>,
    _element: PhantomData<E>,
}
impl<E: WgpuElement, const D: usize> MatmulAutotuneOperationSet<E, D> {
    fn new(lhs: WgpuTensor<E, D>, rhs: WgpuTensor<E, D>, out: WgpuTensor<E, D>) -> Self {
        let m = lhs.shape.dims[D - 2];
        let k = lhs.shape.dims[D - 1];
        let n = rhs.shape.dims[D - 1];
        Self {
            key: AutotuneKey::new("matmul".to_string(), log_mkn_input_key(m, k, n)),
            lhs,
            rhs,
            out,
            _element: PhantomData,
        }
    }
}

impl<E: WgpuElement + Element, const D: usize> AutotuneOperationSet
    for MatmulAutotuneOperationSet<E, D>
{
    fn key(&self) -> AutotuneKey {
        self.key.clone()
    }

    fn autotunables(&self) -> Vec<Box<dyn AutotuneOperation>> {
        let lhs = autotune_tensors(&self.lhs);
        let rhs = autotune_tensors(&self.rhs);
        let out = autotune_tensors(&self.out);

        vec![
            Box::new(MemoryCoalescingMatmulAutotuneOperation::<E, 3>::new(
                lhs.clone(),
                rhs.clone(),
                out.clone(),
            )),
            Box::new(Vec4TilingMatmulAutotuneOperation::<E, 3>::new(
                lhs, rhs, out,
            )),
        ]
    }

    fn fastest(self: Box<Self>, fastest_index: usize) -> Box<dyn AutotuneOperation> {
        match fastest_index {
            0 => Box::new(MemoryCoalescingMatmulAutotuneOperation::<E, D>::new(
                self.lhs, self.rhs, self.out,
            )),
            1 => Box::new(Vec4TilingMatmulAutotuneOperation::<E, D>::new(
                self.lhs, self.rhs, self.out,
            )),
            _ => panic!("Fastest index is out of bound"),
        }
    }
}

pub fn matmul_autotune<E: WgpuElement + Element, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    let client = lhs.client.clone();

    let output = init_matrix_output(&lhs, &rhs);

    let operation_set = Box::new(MatmulAutotuneOperationSet::<E, D>::new(
        lhs,
        rhs,
        output.clone(),
    ));

    client.execute_autotune(operation_set);

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
