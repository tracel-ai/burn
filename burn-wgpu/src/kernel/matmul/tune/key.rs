use burn_tensor::Shape;
use core::fmt::Debug;
use std::{fmt::Display, hash::Hash};

#[derive(Hash, Eq, PartialEq, Debug, Clone)]
/// Autotune key representative of matmul versions
pub struct MatmulAutotuneKey {
    round: bool,     // True when all matmul dims are multiples of 64
    broadcast: bool, // True when there are differences in batch size
    anchored_m: usize,
    anchored_k: usize,
    anchored_n: usize,
}

impl Display for MatmulAutotuneKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(
            format!(
                "Matmul - Round:{:?} Broadcast:{:?} m:{:?} k:{:?} n:{:?}",
                self.round, self.broadcast, self.anchored_m, self.anchored_k, self.anchored_n
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
        for b in 0..D - 2 {
            if lhs_shape.dims[b] != rhs_shape.dims[b] {
                broadcast = true;
            }
        }
        let round = m % 64 == 0 && k % 64 == 0 && n % 64 == 0;
        Self {
            round,
            broadcast,
            anchored_m: anchor(m),
            anchored_k: anchor(k),
            anchored_n: anchor(n),
        }
    }
}

fn anchor(x: usize) -> usize {
    let exp = f32::ceil(f32::log2(x as f32)) as u32;
    2_u32.pow(exp) as usize
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
        let lhs_shape: Shape<3> = [3, 511, 512].into();
        let rhs_shape: Shape<3> = [4, 512, 513].into();
        let key = MatmulAutotuneKey::new(&lhs_shape, &rhs_shape);

        assert!(!key.round);
        assert!(key.broadcast);
        assert!(key.anchored_m == 512);
        assert!(key.anchored_k == 512);
        assert!(key.anchored_n == 1024);
    }
}
