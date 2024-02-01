use crate::kernel::{matmul::MatmulAutotuneKey, reduce::ReduceAutotuneKey};
use burn_compute::tune::AutotuneKey;
use serde::{Deserialize, Serialize};
use std::fmt::Display;

#[cfg(any(feature = "fusion", test))]
use crate::fusion::FusionElemWiseAutotuneKey;

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
/// Key for all autotune-enabled operations
pub enum WgpuAutotuneKey {
    /// Key for matmul operation
    Matmul(MatmulAutotuneKey),
    /// Key for sum_dim operations
    SumDim(ReduceAutotuneKey),
    /// Key for prod_dim operations
    ProdDim(ReduceAutotuneKey),
    /// Key for mean_dim operations
    MeanDim(ReduceAutotuneKey),
    #[cfg(any(feature = "fusion", test))]
    /// Key for fused element wise operations.
    FusionElemWise(FusionElemWiseAutotuneKey),
}

impl Display for WgpuAutotuneKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WgpuAutotuneKey::Matmul(matmul_key) => std::fmt::Display::fmt(&matmul_key, f),
            WgpuAutotuneKey::SumDim(reduce_key) => std::fmt::Display::fmt(&reduce_key, f),
            WgpuAutotuneKey::ProdDim(reduce_key) => std::fmt::Display::fmt(&reduce_key, f),
            WgpuAutotuneKey::MeanDim(reduce_key) => std::fmt::Display::fmt(&reduce_key, f),
            #[cfg(any(feature = "fusion", test))]
            WgpuAutotuneKey::FusionElemWise(reduce_key) => std::fmt::Display::fmt(&reduce_key, f),
        }
    }
}

impl AutotuneKey for WgpuAutotuneKey {}
