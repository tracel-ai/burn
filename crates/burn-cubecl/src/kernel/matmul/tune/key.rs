use crate::{tensor::CubeTensor, CubeAutotuneKey, CubeRuntime, FloatElement};
use burn_tensor::{DType, Shape};
use core::fmt::Debug;
use cubecl::AutotuneKey;
use serde::{Deserialize, Serialize};
use std::{cmp::max, hash::Hash};

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
/// Autotune key representative of matmul versions
pub struct MatmulAutotuneKey {
    round: bool,     // True when all matmul dims are multiples of 64
    broadcast: bool, // True when there are differences in batch size
    #[autotune(anchor)]
    m: usize,
    #[autotune(anchor)]
    k: usize,
    #[autotune(anchor)]
    n: usize,
    #[autotune(anchor(max = 256))]
    batch: usize,
    dtype: DType,
}

impl MatmulAutotuneKey {
    pub(crate) fn from_shape(lhs_shape: &Shape, rhs_shape: &Shape, dtype: DType) -> Self {
        let ndims = lhs_shape.num_dims();
        let m = lhs_shape.dims[ndims - 2];
        let k = lhs_shape.dims[ndims - 1];
        let n = rhs_shape.dims[ndims - 1];

        let mut broadcast = false;
        let mut batch_product_lhs = 1;
        let mut batch_product_rhs = 1;

        for b in 0..ndims - 2 {
            batch_product_lhs *= lhs_shape.dims[b];
            batch_product_rhs *= rhs_shape.dims[b];
            if lhs_shape.dims[b] != rhs_shape.dims[b] {
                broadcast = true;
            }
        }
        let batch_product = max(batch_product_lhs, batch_product_rhs);

        let round = m % 64 == 0 && k % 64 == 0 && n % 64 == 0;

        Self::new(round, broadcast, m, k, n, batch_product, dtype)
    }
}

pub(crate) fn create_key<R: CubeRuntime, E: FloatElement>(
    lhs: &CubeTensor<R>,
    rhs: &CubeTensor<R>,
    _out: &CubeTensor<R>,
) -> CubeAutotuneKey {
    CubeAutotuneKey::Matmul(MatmulAutotuneKey::from_shape(
        &lhs.shape,
        &rhs.shape,
        E::dtype(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_autotune_key_all_same_and_round() {
        let lhs_shape: Shape = [4, 512, 512].into();
        let rhs_shape: Shape = [4, 512, 512].into();
        let key = MatmulAutotuneKey::from_shape(&lhs_shape, &rhs_shape, DType::F32);

        assert!(key.round);
        assert!(!key.broadcast);
        assert_eq!(key.m, 512);
        assert_eq!(key.k, 512);
        assert_eq!(key.n, 512);
    }

    #[test]
    fn matmul_autotune_key_all_different() {
        let lhs_shape: Shape = [2, 3, 511, 512].into();
        let rhs_shape: Shape = [3, 2, 512, 513].into();
        let key = MatmulAutotuneKey::from_shape(&lhs_shape, &rhs_shape, DType::F32);

        assert!(!key.round);
        assert!(key.broadcast);
        assert_eq!(key.m, 512);
        assert_eq!(key.k, 512);
        assert_eq!(key.n, 1024);
        assert_eq!(key.batch, 8);
    }

    #[test]
    fn matmul_autotune_key_large_batch() {
        let lhs_shape: Shape = [128, 512, 511, 512].into();
        let rhs_shape: Shape = [200, 400, 512, 513].into();
        let key = MatmulAutotuneKey::from_shape(&lhs_shape, &rhs_shape, DType::F32);

        assert_eq!(key.batch, 256);
    }
}
