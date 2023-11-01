use burn_compute::tune::{AutotuneKey, AutotuneOperation, AutotuneOperationSet};
use burn_tensor::Element;

use crate::{
    element::WgpuElement,
    kernel::matmul::{tune::utils::autotune_tensors, utils::init_matmul_output},
    tensor::WgpuTensor,
};

/// Set of matmul implementations available for autotune
/// Autotune key is given by concatenating the closest upper power of 2 of m, k and n
pub struct MatmulAutotuneOperationSet<E: WgpuElement, const D: usize> {
    key: AutotuneKey,
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
    out: WgpuTensor<E, D>,
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
        }
    }
}

fn log_mkn_input_key(m: usize, k: usize, n: usize) -> String {
    let mut desc = String::new();
    let mut diff = false;

    for size in [m, k, n] {
        if !desc.is_empty() {
            desc.push('-');
        }
        let exp = f32::ceil(f32::log2(size as f32)) as u32;
        let updated = 2_u32.pow(exp);

        if updated != size as u32 {
            diff = true;
        }
        desc.push_str(updated.to_string().as_str());
    }

    if diff {
        desc.push_str("-uneven");
    }

    desc
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
            Box::new(MemoryCoalescingMatmulDefault::<E, 3>::new(
                lhs.clone(),
                rhs.clone(),
                out.clone(),
            )),
            Box::new(MemoryCoalescingMatmulW16x16::<E, 3>::new(
                lhs.clone(),
                rhs.clone(),
                out.clone(),
            )),
            Box::new(Vec4TilingMatmulDefault::<E, 3>::new(
                lhs.clone(),
                rhs.clone(),
                out.clone(),
            )),
            Box::new(Vec4TilingMatmulUnpaddedDefault::<E, 3>::new(
                lhs.clone(),
                rhs.clone(),
                out.clone(),
            )),
            Box::new(Vec4LhsOnlyTilingMatmulDefault::<E, 3>::new(lhs, rhs, out)),
        ]
    }

    fn fastest(self: Box<Self>, fastest_index: usize) -> Box<dyn AutotuneOperation> {
        match fastest_index {
            0 => Box::new(MemoryCoalescingMatmulDefault::<E, D>::new(
                self.lhs, self.rhs, self.out,
            )),
            1 => Box::new(MemoryCoalescingMatmulW16x16::<E, D>::new(
                self.lhs, self.rhs, self.out,
            )),
            2 => Box::new(Vec4TilingMatmulDefault::<E, D>::new(
                self.lhs, self.rhs, self.out,
            )),
            3 => Box::new(Vec4TilingMatmulUnpaddedDefault::<E, D>::new(
                self.lhs, self.rhs, self.out,
            )),
            4 => Box::new(Vec4LhsOnlyTilingMatmulDefault::<E, D>::new(
                self.lhs, self.rhs, self.out,
            )),
            _ => panic!("Fastest index is out of bound"),
        }
    }
}

/// Executes autotune on matmul operations
pub fn matmul_autotune<E: WgpuElement + Element, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    let client = lhs.client.clone();

    let output = init_matmul_output(&lhs, &rhs);

    let operation_set = Box::new(MatmulAutotuneOperationSet::<E, D>::new(
        lhs,
        rhs,
        output.clone(),
    ));

    client.execute_autotune(operation_set);

    output
}

macro_rules! matmul_tune_ops {
    ($name:ident, $func:expr) => {
        #[derive(new)]
        pub(crate) struct $name<E: WgpuElement, const D: usize> {
            lhs: WgpuTensor<E, D>,
            rhs: WgpuTensor<E, D>,
            out: WgpuTensor<E, D>,
        }

        impl<E: WgpuElement, const D: usize> AutotuneOperation for $name<E, D> {
            fn execute(self: Box<Self>) {
                #[allow(clippy::redundant_closure_call)]
                $func(self.lhs, self.rhs, self.out);
            }

            fn clone(&self) -> Box<dyn AutotuneOperation> {
                Box::new(Self {
                    lhs: self.lhs.clone(),
                    rhs: self.rhs.clone(),
                    out: self.out.clone(),
                })
            }
        }
    };
}

// Potentially better for small matrices.
matmul_tune_ops!(
    MemoryCoalescingMatmulDefault,
    crate::kernel::matmul::matmul_mem_coalescing_default
);

// Potentially better for small matrices.
matmul_tune_ops!(MemoryCoalescingMatmulW16x16, |lhs, rhs, out| {
    crate::kernel::matmul::matmul_mem_coalescing(lhs, rhs, out, 16, 16)
});

// Maybe the fastest on MacOS.
matmul_tune_ops!(
    Vec4LhsOnlyTilingMatmulDefault,
    crate::kernel::matmul::vec4_lhs::matmul_tiling_2d_vec4_lhs
);

// Probably the fastest when fixed sizes.
matmul_tune_ops!(
    Vec4TilingMatmulDefault,
    crate::kernel::matmul::vec4::matmul_tiling_2d_vec4
);

// Probably the fastest otherwise.
matmul_tune_ops!(
    Vec4TilingMatmulUnpaddedDefault,
    crate::kernel::matmul::unpadded::matmul_tiling_2d_unpadded
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_autotune_mkn_key() {
        let key = log_mkn_input_key(512, 512, 512);
        assert_eq!(key, "512-512-512");

        let key = log_mkn_input_key(512, 256, 512);
        assert_eq!(key, "512-256-512");

        let key = log_mkn_input_key(512, 256, 127);
        assert_eq!(key, "512-256-128-uneven");

        let key = log_mkn_input_key(2, 149, 2344);
        assert_eq!(key, "2-256-4096-uneven");
    }
}
