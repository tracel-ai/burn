use crate::kernel::{
    conv::{Conv2dAutotuneKey, ConvTranspose2dAutotuneKey},
    matmul::MatmulAutotuneKey,
    reduce::{ReduceAutotuneKey, SumAutotuneKey},
};
use cubecl::tune::AutotuneKey;
use serde::{Deserialize, Serialize};
use std::fmt::Display;

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
/// Key for all autotune-enabled operations
pub enum CubeAutotuneKey {
    /// Key for matmul operation
    Matmul(MatmulAutotuneKey),
    /// Key for reduce dim operations
    Reduce(ReduceAutotuneKey),
    /// Key for sum operations
    Sum(SumAutotuneKey),
    /// Key for convolution operations
    Conv2d(Conv2dAutotuneKey),
    /// Key for transpose convolution operations
    ConvTranspose2d(ConvTranspose2dAutotuneKey),
}

impl Display for CubeAutotuneKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CubeAutotuneKey::Matmul(matmul_key) => std::fmt::Display::fmt(&matmul_key, f),
            CubeAutotuneKey::Reduce(reduce_key) => std::fmt::Display::fmt(&reduce_key, f),
            CubeAutotuneKey::Sum(reduce_key) => std::fmt::Display::fmt(&reduce_key, f),
            CubeAutotuneKey::Conv2d(conv2d_key) => std::fmt::Display::fmt(&conv2d_key, f),
            CubeAutotuneKey::ConvTranspose2d(conv2d_key) => std::fmt::Display::fmt(&conv2d_key, f),
        }
    }
}

impl AutotuneKey for CubeAutotuneKey {}
