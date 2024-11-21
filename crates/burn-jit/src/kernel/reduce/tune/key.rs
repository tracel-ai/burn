use cubecl::AutotuneKey;
use serde::{Deserialize, Serialize};

use burn_tensor::DType;

use crate::{tensor::JitTensor, JitAutotuneKey, JitElement, JitRuntime};

/// Autotune key representative of reduce versions
#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
pub struct ReduceAutotuneKey {
    #[autotune(anchor)]
    pub(crate) reduce_dim_length: usize,
    #[autotune(anchor)]
    pub(crate) reduce_dim_stride: usize,
    #[autotune(anchor)]
    pub(crate) others_product: usize,
    dtype: DType,
}

pub(crate) fn create_key<R: JitRuntime, EI: JitElement>(
    input: &JitTensor<R>,
    reduce_dim: &usize,
) -> JitAutotuneKey {
    let dims = &input.shape.dims;
    let reduce_dim = *reduce_dim;

    let mut others_product = 1;
    for (d, len) in dims.iter().enumerate() {
        if d != reduce_dim {
            others_product *= len
        }
    }
    JitAutotuneKey::ReduceDim(ReduceAutotuneKey::new(
        dims[reduce_dim],
        input.strides[reduce_dim],
        others_product,
        EI::dtype(),
    ))
}
