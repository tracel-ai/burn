use serde::{Deserialize, Serialize};

/// Controls which operations can be fused.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct FuseSettings {
    /// Enables broadcasting of shapes.
    pub broadcast: bool,
    /// Enables output shape updates.
    ///
    /// When broadcast is enabled, the output shape can become bigger after a fusion,
    /// therefore an update is needed.
    pub output_shape_updates: bool,
    /// Enables the reuse of input buffers.
    pub inplace: bool,
}
