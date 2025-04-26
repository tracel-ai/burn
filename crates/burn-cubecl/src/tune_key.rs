use crate::kernel::{
    conv::{Conv2dAutotuneKey, ConvTranspose2dAutotuneKey},
    reduce::SumAutotuneKey,
};
use cubecl::tune::AutotuneKey;
use serde::{Deserialize, Serialize};
use std::fmt::Display;

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
/// Key for all autotune-enabled operations
pub enum CubeAutotuneKey {
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
            CubeAutotuneKey::Sum(reduce_key) => std::fmt::Debug::fmt(&reduce_key, f),
            CubeAutotuneKey::Conv2d(conv2d_key) => std::fmt::Debug::fmt(&conv2d_key, f),
            CubeAutotuneKey::ConvTranspose2d(conv2d_key) => std::fmt::Debug::fmt(&conv2d_key, f),
        }
    }
}

impl AutotuneKey for CubeAutotuneKey {}
