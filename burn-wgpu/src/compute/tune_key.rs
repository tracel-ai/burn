use serde::{Deserialize, Serialize};
use std::fmt::Display;

use burn_compute::tune::AutotuneKey;

use crate::kernel::{matmul::MatmulAutotuneKey, reduce::ReduceAutotuneKey};

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
/// Key for all autotune-enabled operations
pub enum WgpuAutotuneKey {
    /// Key for matmul operation
    Matmul(MatmulAutotuneKey),
    /// Key for sum_dim operations
    SumDim(ReduceAutotuneKey),
    /// Key for mean_dim operations
    MeanDim(ReduceAutotuneKey),
    ElemWise(()),
}

impl Display for WgpuAutotuneKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WgpuAutotuneKey::Matmul(matmul_key) => std::fmt::Display::fmt(&matmul_key, f),
            WgpuAutotuneKey::SumDim(reduce_key) => std::fmt::Display::fmt(&reduce_key, f),
            WgpuAutotuneKey::MeanDim(reduce_key) => std::fmt::Display::fmt(&reduce_key, f),
            WgpuAutotuneKey::ElemWise(reduce_key) => std::fmt::Display::fmt(&"a", f),
        }
    }
}

impl AutotuneKey for WgpuAutotuneKey {}
