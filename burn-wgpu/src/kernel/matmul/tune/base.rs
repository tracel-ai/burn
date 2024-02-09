use burn_compute::tune::{AutotuneOperation, AutotuneOperationSet};
use burn_tensor::{Element, ElementConversion};

use crate::{
    compute::WgpuAutotuneKey,
    element::WgpuElement,
    kernel::{matmul::utils::init_matmul_output, prng::random_like_uniform},
    ops::numeric::empty_device,
    tensor::WgpuTensor,
    JitRuntime,
};

use super::key::MatmulAutotuneKey;

/// Set of matmul implementations available for autotune
/// Autotune key is given by concatenating the closest upper power of 2 of m, k and n
pub struct MatmulAutotuneOperationSet<B: JitRuntime, E: WgpuElement, const D: usize> {
    key: WgpuAutotuneKey,
    lhs: WgpuTensor<B, E, D>,
    rhs: WgpuTensor<B, E, D>,
    out: WgpuTensor<B, E, D>,
}
impl<B: JitRuntime, E: WgpuElement, const D: usize> MatmulAutotuneOperationSet<B, E, D> {
    fn new(lhs: WgpuTensor<B, E, D>, rhs: WgpuTensor<B, E, D>, out: WgpuTensor<B, E, D>) -> Self {
        Self {
            key: WgpuAutotuneKey::Matmul(MatmulAutotuneKey::new(&lhs.shape, &rhs.shape)),
            lhs,
            rhs,
            out,
        }
    }
}

impl<B: JitRuntime, E: WgpuElement + Element, const D: usize>
    AutotuneOperationSet<WgpuAutotuneKey> for MatmulAutotuneOperationSet<B, E, D>
{
    fn key(&self) -> WgpuAutotuneKey {
        self.key.clone()
    }

    fn autotunables(&self) -> Vec<Box<dyn AutotuneOperation>> {
        let random_bounds: (E, E) = ((-10.0).elem::<E>(), (10.0).elem::<E>());
        let lhs = random_like_uniform(&self.lhs, random_bounds.0, random_bounds.1);
        let rhs = random_like_uniform(&self.rhs, random_bounds.0, random_bounds.1);

        let out = empty_device(
            self.out.client.clone(),
            self.out.device.clone(),
            self.out.shape.clone(),
        );

        vec![
            Box::new(MemoryCoalescingMatmulDefault::new(
                lhs.clone(),
                rhs.clone(),
                out.clone(),
            )),
            Box::new(MemoryCoalescingMatmulW16x16::new(
                lhs.clone(),
                rhs.clone(),
                out.clone(),
            )),
            Box::new(Vec4TilingMatmulDefault::new(
                lhs.clone(),
                rhs.clone(),
                out.clone(),
            )),
            Box::new(Vec4TilingMatmulUnpaddedDefault::new(
                lhs.clone(),
                rhs.clone(),
                out.clone(),
            )),
            Box::new(Vec4LhsOnlyTilingMatmulDefault::new(
                lhs.clone(),
                rhs.clone(),
                out.clone(),
            )),
        ]
    }

    fn fastest(self: Box<Self>, fastest_index: usize) -> Box<dyn AutotuneOperation> {
        match fastest_index {
            0 => Box::new(MemoryCoalescingMatmulDefault::new(
                self.lhs, self.rhs, self.out,
            )),
            1 => Box::new(MemoryCoalescingMatmulW16x16::new(
                self.lhs, self.rhs, self.out,
            )),
            2 => Box::new(Vec4TilingMatmulDefault::new(self.lhs, self.rhs, self.out)),
            3 => Box::new(Vec4TilingMatmulUnpaddedDefault::new(
                self.lhs, self.rhs, self.out,
            )),
            4 => Box::new(Vec4LhsOnlyTilingMatmulDefault::new(
                self.lhs, self.rhs, self.out,
            )),
            _ => panic!("Fastest index is out of bound"),
        }
    }
}

/// Executes autotune on matmul operations
pub fn matmul_autotune<B: JitRuntime, E: WgpuElement + Element, const D: usize>(
    lhs: WgpuTensor<B, E, D>,
    rhs: WgpuTensor<B, E, D>,
) -> WgpuTensor<B, E, D> {
    let client = lhs.client.clone();

    let output = init_matmul_output(&lhs, &rhs);

    let operation_set = Box::new(MatmulAutotuneOperationSet::new(lhs, rhs, output.clone()));

    client.autotune_execute(operation_set);

    output
}

macro_rules! matmul_tune_ops {
    ($name:ident, $func:expr) => {
        #[derive(new)]
        pub(crate) struct $name<B: JitRuntime, E: WgpuElement, const D: usize> {
            lhs: WgpuTensor<B, E, D>,
            rhs: WgpuTensor<B, E, D>,
            out: WgpuTensor<B, E, D>,
        }

        impl<B: JitRuntime, E: WgpuElement, const D: usize> AutotuneOperation
            for $name<B, E, D>
        {
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
