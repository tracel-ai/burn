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
    /// Whether vectorization is enabled.
    pub vectorization: VectorizationSetting,
    /// How [reference layout](super::ir::RefLayout) selection is done.
    pub ref_layout: RefLayoutSetting,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
/// How vectorization is handled during fusion.
pub enum VectorizationSetting {
    /// The biggest line_size possible will be used.
    Activated,
    /// Equivalent to using line_size of one.
    Deactivated,
    /// This is a good setting when a block processes values calculated from a previous block.
    SmallerOrEqualThanPreviousBlock,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
/// Influence how the [reference layout](super::ir::RefLayout) selection is done.
pub enum RefLayoutSetting {
    /// Any reference layout is allowed.
    Any,
    /// Only contiguous reference layout is allowed.
    ///
    /// Note that forcing a contiguous reference layout might reduce the opportunity of inplace
    /// fusion.
    OnlyContiguous,
}
