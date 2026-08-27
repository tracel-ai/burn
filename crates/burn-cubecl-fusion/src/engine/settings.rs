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
    /// Whether the block may write its outputs in a permuted dimension order.
    ///
    /// Writing an output in any dense layout costs the same, so a block whose
    /// inputs are permuted — everything downstream of a convolution, which hands
    /// over an NCHW view of NHWC memory — is better off adopting their order than
    /// reading them strided. Only meaningful together with [RefLayoutSetting::Any];
    /// the settings that constrain the reference already rule a permuted layout out.
    ///
    /// Off by default, because it is only safe for a runner that reads and writes
    /// every operand through the generic fused paths. The matmul runner does not:
    /// it describes its output to the matmul algorithm as row-major while building
    /// the output view from the reference's last two strides, so a permuted
    /// reference makes it write lines that are not contiguous along the column.
    pub choose_output_layout: bool,
}

impl Default for FuseSettings {
    fn default() -> Self {
        Self {
            broadcast: true,
            output_shape_updates: true,
            inplace: true,
            vectorization: VectorizationSetting::Activated,
            ref_layout: RefLayoutSetting::Any,
            choose_output_layout: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
/// How vectorization is handled during fusion.
pub enum VectorizationSetting {
    /// The biggest vector_size possible will be used.
    Activated,
    /// Equivalent to using vector_size of one.
    Deactivated,
    /// This is a good setting when a block processes values calculated from a previous block.
    SmallerOrEqualThanPreviousBlock { block_pos: usize },
    /// This is a good setting when a block processes values calculated from a previous block.
    EqualThanPreviousBlock { block_pos: usize },
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
    SameAsBlock {
        block_pos: u32,
    },
}
