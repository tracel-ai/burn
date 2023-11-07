use std::fmt::Display;

use burn_compute::tune::AutotuneKey;

use crate::kernel::matmul::MatmulAutotuneKey;

#[derive(Hash, Eq, PartialEq, Debug, Clone)]
/// Key for all autotune-enabled operations
pub enum WgpuAutotuneKey {
    /// Key for matmul operation
    Matmul(MatmulAutotuneKey),
}

impl Display for WgpuAutotuneKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WgpuAutotuneKey::Matmul(matmul_key) => std::fmt::Display::fmt(&matmul_key, f),
        }
    }
}

impl AutotuneKey for WgpuAutotuneKey {}
