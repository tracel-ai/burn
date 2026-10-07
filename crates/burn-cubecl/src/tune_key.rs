use crate::kernel::{
    conv::{ConvAutotuneKey, ConvTranspose2dAutotuneKey, ConvTranspose3dAutotuneKey},
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
    Conv(ConvAutotuneKey),
    /// Key for transpose convolution operations
    ConvTranspose(ConvTranspose2dAutotuneKey),
    /// Key for 3D transpose convolution operations
    ConvTranspose3d(ConvTranspose3dAutotuneKey),
}

impl Display for CubeAutotuneKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CubeAutotuneKey::Sum(reduce_key) => std::fmt::Debug::fmt(&reduce_key, f),
            CubeAutotuneKey::Conv(conv_key) => std::fmt::Debug::fmt(&conv_key, f),
            CubeAutotuneKey::ConvTranspose(conv_key) => std::fmt::Debug::fmt(&conv_key, f),
            CubeAutotuneKey::ConvTranspose3d(conv_key) => std::fmt::Debug::fmt(&conv_key, f),
        }
    }
}

impl AutotuneKey for CubeAutotuneKey {}
