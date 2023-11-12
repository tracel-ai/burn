use burn_compute::tune::{AutotuneOperation, AutotuneOperationSet};
use burn_tensor::{Element, ElementConversion};

use crate::{
    compute::WgpuAutotuneKey,
    element::WgpuElement,
    kernel::{matmul::utils::init_matmul_output, prng::random_like_uniform},
    ops::numeric::empty_device,
    tensor::WgpuTensor,
};

use super::key::MatmulAutotuneKey;

/// Set of matmul implementations available for autotune
/// Autotune key is given by concatenating the closest upper power of 2 of m, k and n
pub struct MatmulAutotuneOperationSet<E: WgpuElement, const D: usize> {
    key: WgpuAutotuneKey,
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
    out: WgpuTensor<E, D>,
}
impl<E: WgpuElement, const D: usize> MatmulAutotuneOperationSet<E, D> {
    fn new(lhs: WgpuTensor<E, D>, rhs: WgpuTensor<E, D>, out: WgpuTensor<E, D>) -> Self {
        Self {
            key: WgpuAutotuneKey::Matmul(MatmulAutotuneKey::new(&lhs.shape, &rhs.shape)),
            lhs,
            rhs,
            out,
        }
    }
}

impl<E: WgpuElement + Element, const D: usize> AutotuneOperationSet<WgpuAutotuneKey>
    for MatmulAutotuneOperationSet<E, D>
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
            Box::new(MemoryCoalescingMatmulDefault::<E, D>::new(
                lhs.clone(),
                rhs.clone(),
                out.clone(),
            )),
            Box::new(MemoryCoalescingMatmulW16x16::<E, D>::new(
                lhs.clone(),
                rhs.clone(),
                out.clone(),
            )),
            Box::new(Vec4TilingMatmulDefault::<E, D>::new(
                lhs.clone(),
                rhs.clone(),
                out.clone(),
            )),
            Box::new(Vec4TilingMatmulUnpaddedDefault::<E, D>::new(
                lhs.clone(),
                rhs.clone(),
                out.clone(),
            )),
            Box::new(Vec4LhsOnlyTilingMatmulDefault::<E, D>::new(
                lhs.clone(),
                rhs.clone(),
                out.clone(),
            )),
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
