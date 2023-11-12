use burn_tensor::Shape;
use core::fmt::Debug;
use std::{
    cmp::{max, min},
    fmt::Display,
    hash::Hash,
};

#[derive(Hash, Eq, PartialEq, Debug, Clone)]
/// Autotune key representative of matmul versions
pub struct MatmulAutotuneKey {
    round: bool,     // True when all matmul dims are multiples of 64
    broadcast: bool, // True when there are differences in batch size
    anchored_m: usize,
    anchored_k: usize,
    anchored_n: usize,
    anchored_batch: usize,
}

impl Display for MatmulAutotuneKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(
            format!(
                "Matmul - Round:{:?} Broadcast:{:?} m:{:?} k:{:?} n:{:?} batch:{:?}",
                self.round,
                self.broadcast,
                self.anchored_m,
                self.anchored_k,
                self.anchored_n,
                self.anchored_batch
            )
            .as_str(),
        )
    }
}

impl MatmulAutotuneKey {
    /// Create a matmul autotune key from the input shapes
    pub fn new<const D: usize>(lhs_shape: &Shape<D>, rhs_shape: &Shape<D>) -> Self {
        let m = lhs_shape.dims[D - 2];
        let k = lhs_shape.dims[D - 1];
        let n = rhs_shape.dims[D - 1];

        let mut broadcast = false;
        let mut batch_product_lhs = 1;
        let mut batch_product_rhs = 1;

        for b in 0..D - 2 {
            batch_product_lhs *= lhs_shape.dims[b];
            batch_product_rhs *= rhs_shape.dims[b];
            if lhs_shape.dims[b] != rhs_shape.dims[b] {
                broadcast = true;
            }
        }
        let batch_product = max(batch_product_lhs, batch_product_rhs);

        let round = m % 64 == 0 && k % 64 == 0 && n % 64 == 0;

        Self {
            round,
            broadcast,
            anchored_m: anchor(m, None),
            anchored_k: anchor(k, None),
            anchored_n: anchor(n, None),
            anchored_batch: anchor(batch_product, Some(256)),
        }
    }
}

fn anchor(x: usize, max: Option<usize>) -> usize {
    let exp = f32::ceil(f32::log2(x as f32)) as u32;
    let power_of_2 = 2_u32.pow(exp) as usize;
    if let Some(max) = max {
        min(power_of_2, max)
    } else {
        power_of_2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_autotune_key_all_same_and_round() {
        let lhs_shape: Shape<3> = [4, 512, 512].into();
        let rhs_shape: Shape<3> = [4, 512, 512].into();
        let key = MatmulAutotuneKey::new(&lhs_shape, &rhs_shape);

        assert!(key.round);
        assert!(!key.broadcast);
        assert!(key.anchored_m == 512);
        assert!(key.anchored_k == 512);
        assert!(key.anchored_n == 512);
    }

    #[test]
    fn matmul_autotune_key_all_different() {
        let lhs_shape: Shape<4> = [2, 3, 511, 512].into();
        let rhs_shape: Shape<4> = [3, 2, 512, 513].into();
        let key = MatmulAutotuneKey::new(&lhs_shape, &rhs_shape);

        assert!(!key.round);
        assert!(key.broadcast);
        assert!(key.anchored_m == 512);
        assert!(key.anchored_k == 512);
        assert!(key.anchored_n == 1024);
        assert!(key.anchored_batch == 8);
    }

    #[test]
    fn matmul_autotune_key_large_batch() {
        let lhs_shape: Shape<4> = [128, 512, 511, 512].into();
        let rhs_shape: Shape<4> = [200, 400, 512, 513].into();
        let key = MatmulAutotuneKey::new(&lhs_shape, &rhs_shape);

        assert!(key.anchored_batch == 256);
    }
}
