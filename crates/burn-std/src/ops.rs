//! Configuration types for tensor operations.

use crate::ElementConversion;
use core::num::NonZeroUsize;

/// Check that the parameter value is non-zero.
// NOTE: for now we keep usize but we could refactor the parameters to hold `NonZeroUsize`.
pub(crate) fn check_nonzero(value: usize, msg: &str) -> usize {
    NonZeroUsize::new(value).expect(msg);
    value
}

/// Convolution options.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ConvOptions<const N: usize> {
    /// Stride (non-zero).
    pub stride: [usize; N],

    /// Padding as `(begin, end)` pairs for each spatial dimension.
    pub padding: [(usize, usize); N],

    /// Dilation (non-zero).
    pub dilation: [usize; N],

    /// Groups (non-zero).
    pub groups: usize,
}

impl<const N: usize> ConvOptions<N> {
    /// Constructs a new `ConvOptions`.
    pub fn new(
        stride: [usize; N],
        padding: [usize; N],
        dilation: [usize; N],
        groups: usize,
    ) -> Self {
        Self {
            stride: stride.map(|s| check_nonzero(s, "stride must be non-zero")),
            padding: padding.map(|padding| (padding, padding)),
            dilation: dilation.map(|d| check_nonzero(d, "dilation must be non-zero")),
            groups: check_nonzero(groups, "groups must be non-zero"),
        }
    }

    /// Constructs convolution options with explicit per-side padding.
    pub fn new_with_padding(
        stride: [usize; N],
        padding: [(usize, usize); N],
        dilation: [usize; N],
        groups: usize,
    ) -> Self {
        Self {
            stride: stride.map(|s| check_nonzero(s, "stride must be non-zero")),
            padding,
            dilation: dilation.map(|d| check_nonzero(d, "dilation must be non-zero")),
            groups: check_nonzero(groups, "groups must be non-zero"),
        }
    }

    /// Returns true if padding is asymmetric.
    pub fn is_asymmetric(&self) -> bool {
        self.padding.iter().any(|(begin, end)| begin != end)
    }

    /// Returns the padding at the beginning of every spatial dimension.
    pub fn padding_begin(&self) -> [usize; N] {
        self.padding.map(|(begin, _)| begin)
    }

    /// Returns the padding at the end of every spatial dimension.
    pub fn padding_end(&self) -> [usize; N] {
        self.padding.map(|(_, after)| after)
    }

    /// Returns symmetric padding values.
    ///
    /// # Panics
    /// Panics when any spatial dimension has asymmetric padding.
    pub fn symmetric_padding(&self) -> [usize; N] {
        assert!(
            !self.is_asymmetric(),
            "expected symmetric convolution padding"
        );
        self.padding_begin()
    }
}

/// Deformable convolution options.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct DeformConvOptions<const N: usize> {
    /// Stride (non-zero).
    pub stride: [usize; N],

    /// Padding.
    pub padding: [usize; N],

    /// Dilation (non-zero).
    pub dilation: [usize; N],

    /// Weight Groups (non-zero).
    pub weight_groups: usize,

    /// Offset Groups (non-zero).
    pub offset_groups: usize,
}

impl<const N: usize> DeformConvOptions<N> {
    /// Constructs a new `DeformConvOptions`.
    pub fn new(
        stride: [usize; N],
        padding: [usize; N],
        dilation: [usize; N],
        weight_groups: usize,
        offset_groups: usize,
    ) -> Self {
        Self {
            stride: stride.map(|s| check_nonzero(s, "stride must be non-zero")),
            padding,
            dilation: dilation.map(|d| check_nonzero(d, "dilation must be non-zero")),
            weight_groups: check_nonzero(weight_groups, "weight groups must be non-zero"),
            offset_groups: check_nonzero(offset_groups, "offset groups must be non-zero"),
        }
    }
}

/// Transposed convolution options.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ConvTransposeOptions<const N: usize> {
    /// Stride (non-zero).
    pub stride: [usize; N],

    /// Padding.
    pub padding: [usize; N],

    /// Padding out.
    pub padding_out: [usize; N],

    /// Dilation (non-zero).
    pub dilation: [usize; N],

    /// Groups (non-zero).
    pub groups: usize,
}

impl<const N: usize> ConvTransposeOptions<N> {
    /// Constructs a new `ConvTransposeOptions`.
    pub fn new(
        stride: [usize; N],
        padding: [usize; N],
        padding_out: [usize; N],
        dilation: [usize; N],
        groups: usize,
    ) -> Self {
        Self {
            stride: stride.map(|s| check_nonzero(s, "stride must be non-zero")),
            padding,
            padding_out,
            dilation: dilation.map(|d| check_nonzero(d, "dilation must be non-zero")),
            groups: check_nonzero(groups, "groups must be non-zero"),
        }
    }
}

/// Unfold operation options.
#[derive(Debug, Clone)]
pub struct UnfoldOptions {
    /// The number of positions to slide over the input tensor in each dimension.
    /// A stride of `[1, 1]` will slide the kernel one pixel at a time.
    pub stride: [usize; 2],

    /// The number of zero-padding pixels added to each side of the input tensor in each dimension.
    pub padding: [usize; 2],

    /// The spacing between the blocks (patches) in the original input tensor.
    pub dilation: [usize; 2],
}

impl UnfoldOptions {
    /// Constructs a new `UnfoldOptions`.
    pub fn new(stride: [usize; 2], padding: [usize; 2], dilation: [usize; 2]) -> Self {
        Self {
            stride: stride.map(|s| check_nonzero(s, "stride must be non-zero")),
            padding,
            dilation: dilation.map(|d| check_nonzero(d, "dilation must be non-zero")),
        }
    }
}

/// Algorithm used.
#[derive(new, Debug, Clone, serde::Deserialize, serde::Serialize)]
pub enum InterpolateMode {
    /// Nearest-neighbor floor interpolation.
    /// Matches the legacy behavior of OpenCV’s INTER_NEAREST. It results in a bottom-right shift when resizing.
    /// <https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation>
    Nearest,

    /// Nearest-neighbor exact interpolation.
    /// <https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation>
    NearestExact,

    /// Bilinear interpolation.
    /// <https://en.wikipedia.org/wiki/Bilinear_interpolation>
    Bilinear,

    /// Bicubic interpolation.
    /// <https://en.wikipedia.org/wiki/Bicubic_interpolation>
    Bicubic,

    /// Lanczos3 interpolation (6-tap sinc-based filter).
    /// <https://en.wikipedia.org/wiki/Lanczos_resampling>
    Lanczos3,
}

/// Interpolation options.
#[derive(Debug, Clone)]
pub struct InterpolateOptions {
    /// Algorithm used.
    pub mode: InterpolateMode,
    /// If `true`, the input and output tensors are aligned by their corner pixels.
    /// If `false`, half-pixel coordinate mapping is used instead.
    pub align_corners: bool,
}

impl InterpolateOptions {
    /// Create new interpolate options with the given mode.
    /// Defaults to `align_corners = true`.
    pub fn new(mode: InterpolateMode) -> Self {
        Self {
            mode,
            align_corners: true,
        }
    }

    /// Set align_corners.
    pub fn with_align_corners(mut self, align_corners: bool) -> Self {
        self.align_corners = align_corners;
        self
    }
}

/// Padding mode for grid sampling when coordinates are out of bounds.
///
/// Matches PyTorch's `padding_mode` parameter in `grid_sample`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize, serde::Serialize)]
pub enum GridSamplePaddingMode {
    /// Fill with zeros for out-of-bounds coordinates.
    #[default]
    Zeros,
    /// Clamp coordinates to the border (use nearest edge value).
    Border,
    /// Reflect coordinates at the boundary.
    Reflection,
}

/// Options for grid sampling operations.
#[derive(Debug, Clone)]
pub struct GridSampleOptions {
    /// Interpolation mode (bilinear, nearest, or bicubic).
    pub mode: InterpolateMode,
    /// Padding mode for out-of-bounds coordinates.
    pub padding_mode: GridSamplePaddingMode,
    /// If `true`, grid values of -1 and 1 correspond to the corner pixels.
    /// If `false`, they correspond to the corner points of the corner pixels
    /// (i.e., -1 maps to -0.5 and 1 maps to size - 0.5 in pixel coordinates).
    pub align_corners: bool,
}

impl Default for GridSampleOptions {
    fn default() -> Self {
        Self {
            mode: InterpolateMode::Bilinear,
            padding_mode: GridSamplePaddingMode::Zeros,
            align_corners: false,
        }
    }
}

impl From<InterpolateMode> for GridSampleOptions {
    fn from(value: InterpolateMode) -> Self {
        GridSampleOptions::new(value)
    }
}

impl GridSampleOptions {
    /// Create new grid sample options with the given interpolation mode.
    ///
    /// Uses default values for padding_mode (Zeros) and align_corners (false).
    pub fn new(mode: InterpolateMode) -> Self {
        Self {
            mode,
            ..Default::default()
        }
    }

    /// Set the padding mode.
    pub fn with_padding_mode(mut self, padding_mode: GridSamplePaddingMode) -> Self {
        self.padding_mode = padding_mode;
        self
    }

    /// Set align_corners.
    pub fn with_align_corners(mut self, align_corners: bool) -> Self {
        self.align_corners = align_corners;
        self
    }
}

/// Padding mode for tensor pad operations.
///
/// Defines how values are filled when padding a tensor beyond its original boundaries.
/// Padding can be applied to any dimension of a tensor.
///
/// # Modes
///
/// - [`Constant`](PadMode::Constant): Fill with a specified value (default: 0.0)
/// - [`Reflect`](PadMode::Reflect): Mirror values at boundary, excluding edge (requires padding < dim_size)
/// - [`Edge`](PadMode::Edge): Replicate boundary values
#[derive(Debug, Clone, Copy, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum PadMode {
    /// Fill padded regions with a constant value.
    ///
    /// # Example
    /// For tensor `[1, 2, 3]` with padding 2 on the left and value 0:
    /// Result: `[0, 0, 1, 2, 3]`
    Constant(f32),

    /// Reflect values at the boundary, excluding the edge value.
    ///
    /// Padding must be less than the dimension size (i.e., `padding < dim_size`).
    ///
    /// # Example
    /// For tensor `[1, 2, 3, 4]` with padding 2 on the left:
    /// Result: `[3, 2, 1, 2, 3, 4]` (reflects from index 1, not 0)
    Reflect,

    /// Replicate the edge values.
    ///
    /// # Example
    /// For tensor `[1, 2, 3, 4]` with padding 2 on the left:
    /// Result: `[1, 1, 1, 2, 3, 4]`
    Edge,
}

impl Default for PadMode {
    fn default() -> Self {
        PadMode::Constant(0.0)
    }
}

impl<E: ElementConversion> From<E> for PadMode {
    fn from(value: E) -> Self {
        PadMode::Constant(value.elem())
    }
}

/// Options for the attention module.
#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct AttentionModuleOptions {
    /// Custom scale factor applied to QK^T. When `None`, defaults to `1/sqrt(head_dim)`.
    pub scale: Option<f64>,

    /// Soft capping applied before softmax: `softcap * tanh(scores / softcap)`.
    /// Used by Gemma-2 and similar models. Must be positive when set.
    pub softcap: Option<f64>,

    /// When `true`, applies causal (autoregressive) masking so that each query position
    /// can only attend to key positions at or before it. This is more efficient than
    /// passing an explicit lower-triangular bool mask because backends can use optimized
    /// kernel paths (e.g. flash attention with causal mode).
    pub is_causal: bool,
}

/// Computation to be used to update the existing values in indexed assignment operations (scatter/select).
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum IndexingUpdateOp {
    /// Overwrite existing values.
    Assign,
    /// Performs an addition.
    Add,
    /// Multiply existing values.
    Mul,
    /// Take element-wise minimum.
    Min,
    /// Take element-wise maximum.
    Max,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv_options_symmetric_constructor() {
        let options = ConvOptions::new([1, 2], [3, 4], [5, 6], 7);

        assert_eq!(options.padding, [(3, 3), (4, 4)]);
        assert_eq!(options.padding_begin(), [3, 4]);
        assert_eq!(options.padding_end(), [3, 4]);
        assert_eq!(options.symmetric_padding(), [3, 4]);
        assert!(!options.is_asymmetric());
    }

    #[test]
    fn conv_options_explicit_padding_constructor() {
        let options = ConvOptions::new_with_padding([1, 2], [(3, 4), (5, 6)], [7, 8], 9);

        assert_eq!(options.padding, [(3, 4), (5, 6)]);
        assert_eq!(options.padding_begin(), [3, 5]);
        assert_eq!(options.padding_end(), [4, 6]);
        assert!(options.is_asymmetric());
    }

    #[test]
    #[should_panic = "expected symmetric convolution padding"]
    fn conv_options_symmetric_padding_with_asymmetric_options() {
        let options = ConvOptions::new_with_padding([1], [(1, 2)], [1], 1);
        let _ = options.symmetric_padding();
    }

    #[test]
    #[should_panic = "stride must be non-zero"]
    fn conv_options_stride_zero() {
        let _opt = ConvOptions::new([0, 1], [0, 0], [1, 1], 1);
    }

    #[test]
    #[should_panic = "dilation must be non-zero"]
    fn conv_options_dilation_zero() {
        let _opt = ConvOptions::new([1, 1], [0, 0], [0, 0], 1);
    }

    #[test]
    #[should_panic = "groups must be non-zero"]
    fn conv_options_groups_zero() {
        let _opt = ConvOptions::new([1, 1], [0, 0], [1, 1], 0);
    }

    #[test]
    #[should_panic = "stride must be non-zero"]
    fn conv_transpose_options_stride_zero() {
        let _opt = ConvTransposeOptions::new([0, 1], [0, 0], [0, 0], [1, 1], 1);
    }

    #[test]
    #[should_panic = "dilation must be non-zero"]
    fn conv_transpose_options_dilation_zero() {
        let _opt = ConvTransposeOptions::new([1, 1], [0, 0], [0, 0], [0, 0], 1);
    }

    #[test]
    #[should_panic = "groups must be non-zero"]
    fn conv_transpose_options_groups_zero() {
        let _opt = ConvTransposeOptions::new([1, 1], [0, 0], [0, 0], [1, 1], 0);
    }

    #[test]
    #[should_panic = "stride must be non-zero"]
    fn deform_conv_options_stride_zero() {
        let _opt = DeformConvOptions::new([0, 1], [0, 0], [1, 1], 1, 1);
    }

    #[test]
    #[should_panic = "dilation must be non-zero"]
    fn deform_conv_options_dilation_zero() {
        let _opt = DeformConvOptions::new([1, 1], [0, 0], [0, 0], 1, 1);
    }

    #[test]
    #[should_panic = "weight groups must be non-zero"]
    fn deform_conv_options_weights_groups_zero() {
        let _opt = DeformConvOptions::new([1, 1], [0, 0], [1, 1], 0, 1);
    }

    #[test]
    #[should_panic = "offset groups must be non-zero"]
    fn deform_conv_options_offset_groups_zero() {
        let _opt = DeformConvOptions::new([1, 1], [0, 0], [1, 1], 1, 0);
    }

    #[test]
    #[should_panic = "stride must be non-zero"]
    fn unfold_options_stride_zero() {
        let _opt = UnfoldOptions::new([0, 1], [0, 0], [1, 1]);
    }

    #[test]
    #[should_panic = "dilation must be non-zero"]
    fn unfold_options_dilation_zero() {
        let _opt = UnfoldOptions::new([1, 1], [0, 0], [0, 0]);
    }
}
