use crate::kernel::{matmul::MatmulAutotuneKey, reduce::ReduceAutotuneKey};
use burn_compute::tune::AutotuneKey;
use serde::{Deserialize, Serialize};
use std::fmt::Display;

#[cfg(any(feature = "fusion", test))]
use crate::fusion::FusionElemWiseAutotuneKey;

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
/// Key for all autotune-enabled operations
pub enum JitAutotuneKey {
    /// Key for matmul operation
    Matmul(MatmulAutotuneKey),
    /// Key for reduce dim operations
    ReduceDim(ReduceAutotuneKey),
    #[cfg(any(feature = "fusion", test))]
    /// Key for fused element wise operations.
    FusionElemWise(FusionElemWiseAutotuneKey),
}

impl Display for JitAutotuneKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JitAutotuneKey::Matmul(matmul_key) => std::fmt::Display::fmt(&matmul_key, f),
            JitAutotuneKey::ReduceDim(reduce_key) => std::fmt::Display::fmt(&reduce_key, f),
            #[cfg(any(feature = "fusion", test))]
            JitAutotuneKey::FusionElemWise(reduce_key) => std::fmt::Display::fmt(&reduce_key, f),
        }
    }
}

impl AutotuneKey for JitAutotuneKey {}
