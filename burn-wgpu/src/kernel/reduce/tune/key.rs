use std::{cmp::min, fmt::Display};

use burn_tensor::Shape;

#[derive(Hash, Eq, PartialEq, Debug, Clone)]
/// Autotune key representative of reduce versions
pub struct ReduceAutotuneKey {
    reduce_dim_length: usize,
    others_product: usize,
}

impl Display for ReduceAutotuneKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(
            format!(
                "Reduce - reduce_dim_length: {:?} others_product: {:?}",
                self.reduce_dim_length, self.others_product
            )
            .as_str(),
        )
    }
}

impl ReduceAutotuneKey {
    /// Create a reduce autotune key from the input shape and reduce dim
    pub fn new<const D: usize>(shape: &Shape<D>, reduce_dim: usize) -> Self {
        let reduce_dim_length = shape.dims[reduce_dim];
        let mut others_product = 1;
        for d in 0..D {
            if d != reduce_dim {
                others_product *= shape.dims[d]
            }
        }
        Self {
            reduce_dim_length: anchor(reduce_dim_length, None),
            others_product: anchor(others_product, None),
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
