use burn_core as burn;

use burn::config::Config;

/// Calculate asymmetric padding for "same" convolution.
/// Returns (start_padding, end_padding) where start is applied first (top/left).
/// For odd total padding, the extra pad goes to the end (bottom/right) following ONNX convention.
fn calculate_same_padding(kernel_size: usize, stride: usize, size_in: usize) -> (usize, usize) {
    let size_out = size_in.div_ceil(stride); // ceil division for same padding
    let total_padding = if size_out > 0 {
        let needed = (size_out - 1) * stride + kernel_size;
        needed.saturating_sub(size_in)
    } else {
        0
    };
    let pad_start = total_padding / 2;
    let pad_end = total_padding - pad_start;
    (pad_start, pad_end)
}

/// Padding configuration for 1D operators.
#[derive(Config, Debug, PartialEq)]
pub enum PaddingConfig1d {
    /// Dynamically calculates padding to ensure output size matches input size.
    Same,
    /// No padding applied.
    Valid,
    /// Applies explicit padding values.
    /// Format: (left, right)
    /// For symmetric padding, use the same value for both (e.g., `Explicit(1, 1)`).
    Explicit(usize, usize),
}

impl PaddingConfig1d {
    /// Calculate padding as (left, right) pair for 1D operations.
    /// For `Same` padding, this computes the actual asymmetric padding if needed.
    pub(crate) fn calculate_padding_1d_pair(
        &self,
        length: usize,
        kernel_size: usize,
        stride: usize,
    ) -> (usize, usize) {
        match self {
            Self::Valid => (0, 0),
            Self::Same => calculate_same_padding(kernel_size, stride, length),
            Self::Explicit(left, right) => (*left, *right),
        }
    }
}

/// Padding configuration for 2D operators.
#[derive(Config, Debug, PartialEq)]
pub enum PaddingConfig2d {
    /// Dynamically calculates padding to preserve input dimensions in output.
    Same,
    /// No padding applied.
    Valid,
    /// Applies explicit padding values.
    /// Format: (top, left, bottom, right)
    /// For symmetric padding, use matching values (e.g., `Explicit(1, 1, 1, 1)`).
    Explicit(usize, usize, usize, usize),
}

impl PaddingConfig2d {
    /// Calculate padding as ((top, bottom), (left, right)) pairs for 2D operations.
    /// For `Same` padding, this computes the actual asymmetric padding if needed.
    pub(crate) fn calculate_padding_2d_pairs(
        &self,
        height: usize,
        width: usize,
        kernel_size: &[usize; 2],
        stride: &[usize; 2],
    ) -> ((usize, usize), (usize, usize)) {
        match self {
            Self::Valid => ((0, 0), (0, 0)),
            Self::Same => {
                let (top, bottom) = calculate_same_padding(kernel_size[0], stride[0], height);
                let (left, right) = calculate_same_padding(kernel_size[1], stride[1], width);
                ((top, bottom), (left, right))
            }
            Self::Explicit(top, left, bottom, right) => ((*top, *bottom), (*left, *right)),
        }
    }

    /// Calculate symmetric padding for 2D operations.
    /// Returns padding values [height, width] (same for both sides).
    /// Panics if asymmetric padding is detected.
    pub(crate) fn calculate_padding_2d(
        &self,
        height: usize,
        width: usize,
        kernel_size: &[usize; 2],
        stride: &[usize; 2],
    ) -> [usize; 2] {
        let ((top, bottom), (left, right)) =
            self.calculate_padding_2d_pairs(height, width, kernel_size, stride);
        if top != bottom || left != right {
            panic!("Asymmetric padding should be handled via calculate_padding_2d_pairs()")
        }
        [top, left]
    }
}

/// Padding configuration for 3D operators.
#[derive(Config, Debug, PartialEq)]
pub enum PaddingConfig3d {
    /// Dynamically calculates padding to preserve input dimensions in output.
    Same,
    /// No padding applied.
    Valid,
    /// Applies explicit symmetric padding values.
    /// Format: (depth, height, width) — same padding on both sides of each dimension.
    Explicit(usize, usize, usize),
}

impl PaddingConfig3d {
    /// Calculate symmetric padding for 3D operations.
    /// Returns padding values [depth, height, width] (same for both sides).
    pub(crate) fn calculate_padding_3d(
        &self,
        depth: usize,
        height: usize,
        width: usize,
        kernel_size: &[usize; 3],
        stride: &[usize; 3],
    ) -> [usize; 3] {
        match self {
            Self::Valid => [0, 0, 0],
            Self::Same => {
                let (front, back) = calculate_same_padding(kernel_size[0], stride[0], depth);
                let (top, bottom) = calculate_same_padding(kernel_size[1], stride[1], height);
                let (left, right) = calculate_same_padding(kernel_size[2], stride[2], width);
                if front != back || top != bottom || left != right {
                    panic!(
                        "Asymmetric 3D 'Same' padding is not supported. \
                        Use odd kernel sizes for symmetric padding."
                    )
                }
                [front, top, left]
            }
            Self::Explicit(depth, height, width) => [*depth, *height, *width],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== PaddingConfig1d Tests ====================

    #[test]
    fn test_padding_config_1d_calculate_pair_valid() {
        let padding = PaddingConfig1d::Valid;
        assert_eq!(padding.calculate_padding_1d_pair(10, 3, 1), (0, 0));
    }

    #[test]
    fn test_padding_config_1d_calculate_pair_explicit() {
        let padding = PaddingConfig1d::Explicit(1, 2);
        assert_eq!(padding.calculate_padding_1d_pair(10, 3, 1), (1, 2));
    }

    #[test]
    fn test_padding_config_1d_calculate_pair_same() {
        let padding = PaddingConfig1d::Same;
        // kernel=3, stride=1, length=10: total=2, start=1, end=1
        assert_eq!(padding.calculate_padding_1d_pair(10, 3, 1), (1, 1));
    }

    // ==================== PaddingConfig2d Tests ====================

    #[test]
    fn test_padding_config_2d_calculate_pairs_valid() {
        let padding = PaddingConfig2d::Valid;
        assert_eq!(
            padding.calculate_padding_2d_pairs(10, 10, &[3, 3], &[1, 1]),
            ((0, 0), (0, 0))
        );
    }

    #[test]
    fn test_padding_config_2d_calculate_pairs_explicit() {
        let padding = PaddingConfig2d::Explicit(1, 2, 3, 4);
        assert_eq!(
            padding.calculate_padding_2d_pairs(10, 10, &[3, 3], &[1, 1]),
            ((1, 3), (2, 4))
        );
    }

    #[test]
    fn test_padding_config_2d_calculate_symmetric_valid() {
        let padding = PaddingConfig2d::Valid;
        assert_eq!(
            padding.calculate_padding_2d(10, 10, &[3, 3], &[1, 1]),
            [0, 0]
        );
    }

    #[test]
    fn test_padding_config_2d_calculate_symmetric_explicit() {
        let padding = PaddingConfig2d::Explicit(2, 3, 2, 3);
        assert_eq!(
            padding.calculate_padding_2d(10, 10, &[3, 3], &[1, 1]),
            [2, 3]
        );
    }

    #[test]
    #[should_panic(
        expected = "Asymmetric padding should be handled via calculate_padding_2d_pairs"
    )]
    fn test_padding_config_2d_calculate_symmetric_asymmetric_panics() {
        let padding = PaddingConfig2d::Explicit(1, 2, 3, 4);
        let _ = padding.calculate_padding_2d(10, 10, &[3, 3], &[1, 1]);
    }

    // ==================== PaddingConfig3d Tests ====================

    #[test]
    fn test_padding_config_3d_calculate_valid() {
        let padding = PaddingConfig3d::Valid;
        assert_eq!(
            padding.calculate_padding_3d(10, 10, 10, &[3, 3, 3], &[1, 1, 1]),
            [0, 0, 0]
        );
    }

    #[test]
    fn test_padding_config_3d_calculate_explicit() {
        let padding = PaddingConfig3d::Explicit(1, 2, 3);
        assert_eq!(
            padding.calculate_padding_3d(10, 10, 10, &[3, 3, 3], &[1, 1, 1]),
            [1, 2, 3]
        );
    }

    #[test]
    fn test_padding_config_3d_calculate_same_odd_kernel() {
        let padding = PaddingConfig3d::Same;
        // kernel=3, stride=1: total=2, symmetric (1,1) per dim
        assert_eq!(
            padding.calculate_padding_3d(10, 10, 10, &[3, 3, 3], &[1, 1, 1]),
            [1, 1, 1]
        );
    }
}
