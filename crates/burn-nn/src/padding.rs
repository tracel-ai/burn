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
    pub fn calculate_padding_1d_pair(
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

    /// Calculate symmetric padding for 1D operations.
    /// Returns a single padding value (same for both sides).
    /// Panics if asymmetric padding is detected.
    #[allow(dead_code)] // Used in tests; kept for API consistency with 2D/3D
    pub(crate) fn calculate_padding_1d(
        &self,
        length: usize,
        kernel_size: usize,
        stride: usize,
    ) -> usize {
        let (left, right) = self.calculate_padding_1d_pair(length, kernel_size, stride);
        if left != right {
            panic!("Asymmetric padding should be handled via calculate_padding_1d_pair()")
        }
        left
    }

    /// Returns true if this padding configuration would produce asymmetric padding.
    /// For `Explicit`, checks if left != right.
    /// For `Same`, this always returns false since actual asymmetry depends on dimensions
    /// and should be checked after calling `calculate_padding_1d_pair()`.
    pub fn is_asymmetric(&self) -> bool {
        match self {
            Self::Explicit(left, right) => left != right,
            _ => false,
        }
    }

    /// Returns the explicit padding values (left, right).
    /// Panics if not Explicit padding. For computed padding, use `calculate_padding_1d_pair()`.
    pub fn as_tuple(&self) -> (usize, usize) {
        match self {
            Self::Explicit(left, right) => (*left, *right),
            _ => panic!("as_tuple() only works with Explicit padding"),
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
    pub fn calculate_padding_2d_pairs(
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

    /// Returns true if this padding configuration would produce asymmetric padding.
    /// For `Explicit`, checks if top != bottom or left != right.
    /// For `Same`, this always returns false since actual asymmetry depends on dimensions
    /// and should be checked after calling `calculate_padding_2d_pairs()`.
    pub fn is_asymmetric(&self) -> bool {
        match self {
            Self::Explicit(top, left, bottom, right) => top != bottom || left != right,
            _ => false,
        }
    }

    /// Returns the explicit padding values (top, left, bottom, right).
    /// Panics if not Explicit padding. For computed padding, use `calculate_padding_2d_pairs()`.
    pub fn as_tuple(&self) -> (usize, usize, usize, usize) {
        match self {
            Self::Explicit(top, left, bottom, right) => (*top, *left, *bottom, *right),
            _ => panic!("as_tuple() only works with Explicit padding"),
        }
    }
}

/// Padding configuration for 3D operators.
#[derive(Config, Debug, PartialEq)]
pub enum PaddingConfig3d {
    /// Dynamically calculates padding to preserve input dimensions in output.
    Same,
    /// No padding applied.
    Valid,
    /// Applies explicit padding values.
    /// Format: (front, top, left, back, bottom, right)
    /// For symmetric padding, use matching values (e.g., `Explicit(1, 1, 1, 1, 1, 1)`).
    Explicit(usize, usize, usize, usize, usize, usize),
}

impl PaddingConfig3d {
    /// Calculate padding as ((front, back), (top, bottom), (left, right)) pairs for 3D operations.
    /// For `Same` padding, this computes the actual asymmetric padding if needed.
    pub fn calculate_padding_3d_pairs(
        &self,
        depth: usize,
        height: usize,
        width: usize,
        kernel_size: &[usize; 3],
        stride: &[usize; 3],
    ) -> ((usize, usize), (usize, usize), (usize, usize)) {
        match self {
            Self::Valid => ((0, 0), (0, 0), (0, 0)),
            Self::Same => {
                let (front, back) = calculate_same_padding(kernel_size[0], stride[0], depth);
                let (top, bottom) = calculate_same_padding(kernel_size[1], stride[1], height);
                let (left, right) = calculate_same_padding(kernel_size[2], stride[2], width);
                ((front, back), (top, bottom), (left, right))
            }
            Self::Explicit(front, top, left, back, bottom, right) => {
                ((*front, *back), (*top, *bottom), (*left, *right))
            }
        }
    }

    /// Calculate symmetric padding for 3D operations.
    /// Returns padding values [depth, height, width] (same for both sides).
    /// Panics if asymmetric padding is detected.
    pub(crate) fn calculate_padding_3d(
        &self,
        depth: usize,
        height: usize,
        width: usize,
        kernel_size: &[usize; 3],
        stride: &[usize; 3],
    ) -> [usize; 3] {
        let ((front, back), (top, bottom), (left, right)) =
            self.calculate_padding_3d_pairs(depth, height, width, kernel_size, stride);
        if front != back || top != bottom || left != right {
            panic!("Asymmetric padding should be handled via calculate_padding_3d_pairs()")
        }
        [front, top, left]
    }

    /// Returns true if this padding configuration would produce asymmetric padding.
    /// For `Explicit`, checks if any dimension has different start/end padding.
    /// For `Same`, this always returns false since actual asymmetry depends on dimensions
    /// and should be checked after calling `calculate_padding_3d_pairs()`.
    pub fn is_asymmetric(&self) -> bool {
        match self {
            Self::Explicit(front, top, left, back, bottom, right) => {
                front != back || top != bottom || left != right
            }
            _ => false,
        }
    }

    /// Returns the explicit padding values (front, top, left, back, bottom, right).
    /// Panics if not Explicit padding. For computed padding, use `calculate_padding_3d_pairs()`.
    pub fn as_tuple(&self) -> (usize, usize, usize, usize, usize, usize) {
        match self {
            Self::Explicit(front, top, left, back, bottom, right) => {
                (*front, *top, *left, *back, *bottom, *right)
            }
            _ => panic!("as_tuple() only works with Explicit padding"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== PaddingConfig1d Tests ====================

    #[test]
    fn test_padding_config_1d_is_asymmetric_symmetric() {
        // Symmetric padding (left == right) should return false
        let padding = PaddingConfig1d::Explicit(2, 2);
        assert!(!padding.is_asymmetric());
    }

    #[test]
    fn test_padding_config_1d_is_asymmetric_asymmetric() {
        // Asymmetric padding (left != right) should return true
        let padding = PaddingConfig1d::Explicit(1, 2);
        assert!(padding.is_asymmetric());
    }

    #[test]
    fn test_padding_config_1d_is_asymmetric_valid() {
        // Valid padding should return false
        let padding = PaddingConfig1d::Valid;
        assert!(!padding.is_asymmetric());
    }

    #[test]
    fn test_padding_config_1d_is_asymmetric_same() {
        // Same padding should return false
        let padding = PaddingConfig1d::Same;
        assert!(!padding.is_asymmetric());
    }

    #[test]
    fn test_padding_config_1d_as_tuple() {
        let padding = PaddingConfig1d::Explicit(1, 2);
        assert_eq!(padding.as_tuple(), (1, 2));
    }

    #[test]
    #[should_panic(expected = "as_tuple() only works with Explicit padding")]
    fn test_padding_config_1d_as_tuple_valid_panics() {
        let padding = PaddingConfig1d::Valid;
        let _ = padding.as_tuple();
    }

    #[test]
    #[should_panic(expected = "as_tuple() only works with Explicit padding")]
    fn test_padding_config_1d_as_tuple_same_panics() {
        let padding = PaddingConfig1d::Same;
        let _ = padding.as_tuple();
    }

    #[test]
    fn test_padding_config_1d_calculate_valid() {
        let padding = PaddingConfig1d::Valid;
        assert_eq!(padding.calculate_padding_1d(10, 3, 1), 0);
    }

    #[test]
    fn test_padding_config_1d_calculate_explicit_symmetric() {
        let padding = PaddingConfig1d::Explicit(2, 2);
        assert_eq!(padding.calculate_padding_1d(10, 3, 1), 2);
    }

    #[test]
    #[should_panic(expected = "Asymmetric padding should be handled via calculate_padding_1d_pair")]
    fn test_padding_config_1d_calculate_explicit_asymmetric_panics() {
        let padding = PaddingConfig1d::Explicit(1, 2);
        let _ = padding.calculate_padding_1d(10, 3, 1);
    }

    // ==================== PaddingConfig2d Tests ====================

    #[test]
    fn test_padding_config_2d_is_asymmetric_symmetric() {
        // Symmetric padding should return false
        let padding = PaddingConfig2d::Explicit(2, 2, 2, 2);
        assert!(!padding.is_asymmetric());
    }

    #[test]
    fn test_padding_config_2d_is_asymmetric_top_bottom() {
        // top != bottom should return true
        let padding = PaddingConfig2d::Explicit(1, 2, 3, 2);
        assert!(padding.is_asymmetric());
    }

    #[test]
    fn test_padding_config_2d_is_asymmetric_left_right() {
        // left != right should return true
        let padding = PaddingConfig2d::Explicit(2, 1, 2, 3);
        assert!(padding.is_asymmetric());
    }

    #[test]
    fn test_padding_config_2d_is_asymmetric_all_different() {
        // All different values should return true
        let padding = PaddingConfig2d::Explicit(1, 2, 3, 4);
        assert!(padding.is_asymmetric());
    }

    #[test]
    fn test_padding_config_2d_is_asymmetric_valid() {
        let padding = PaddingConfig2d::Valid;
        assert!(!padding.is_asymmetric());
    }

    #[test]
    fn test_padding_config_2d_is_asymmetric_same() {
        let padding = PaddingConfig2d::Same;
        assert!(!padding.is_asymmetric());
    }

    #[test]
    fn test_padding_config_2d_as_tuple() {
        let padding = PaddingConfig2d::Explicit(1, 2, 3, 4);
        assert_eq!(padding.as_tuple(), (1, 2, 3, 4));
    }

    #[test]
    #[should_panic(expected = "as_tuple() only works with Explicit padding")]
    fn test_padding_config_2d_as_tuple_valid_panics() {
        let padding = PaddingConfig2d::Valid;
        let _ = padding.as_tuple();
    }

    #[test]
    #[should_panic(expected = "as_tuple() only works with Explicit padding")]
    fn test_padding_config_2d_as_tuple_same_panics() {
        let padding = PaddingConfig2d::Same;
        let _ = padding.as_tuple();
    }

    #[test]
    fn test_padding_config_2d_calculate_valid() {
        let padding = PaddingConfig2d::Valid;
        assert_eq!(
            padding.calculate_padding_2d(10, 10, &[3, 3], &[1, 1]),
            [0, 0]
        );
    }

    #[test]
    fn test_padding_config_2d_calculate_explicit_symmetric() {
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
    fn test_padding_config_2d_calculate_explicit_asymmetric_panics() {
        let padding = PaddingConfig2d::Explicit(1, 2, 3, 4);
        let _ = padding.calculate_padding_2d(10, 10, &[3, 3], &[1, 1]);
    }

    // ==================== PaddingConfig3d Tests ====================

    #[test]
    fn test_padding_config_3d_is_asymmetric_symmetric() {
        // Symmetric padding should return false
        let padding = PaddingConfig3d::Explicit(2, 3, 1, 2, 3, 1);
        assert!(!padding.is_asymmetric());
    }

    #[test]
    fn test_padding_config_3d_is_asymmetric_front_back() {
        // front != back should return true
        let padding = PaddingConfig3d::Explicit(1, 3, 1, 2, 3, 1);
        assert!(padding.is_asymmetric());
    }

    #[test]
    fn test_padding_config_3d_is_asymmetric_top_bottom() {
        // top != bottom should return true
        let padding = PaddingConfig3d::Explicit(2, 1, 1, 2, 3, 1);
        assert!(padding.is_asymmetric());
    }

    #[test]
    fn test_padding_config_3d_is_asymmetric_left_right() {
        // left != right should return true
        let padding = PaddingConfig3d::Explicit(2, 3, 1, 2, 3, 4);
        assert!(padding.is_asymmetric());
    }

    #[test]
    fn test_padding_config_3d_is_asymmetric_all_different() {
        // All different values should return true
        let padding = PaddingConfig3d::Explicit(1, 2, 3, 4, 5, 6);
        assert!(padding.is_asymmetric());
    }

    #[test]
    fn test_padding_config_3d_is_asymmetric_valid() {
        let padding = PaddingConfig3d::Valid;
        assert!(!padding.is_asymmetric());
    }

    #[test]
    fn test_padding_config_3d_is_asymmetric_same() {
        let padding = PaddingConfig3d::Same;
        assert!(!padding.is_asymmetric());
    }

    #[test]
    fn test_padding_config_3d_as_tuple() {
        let padding = PaddingConfig3d::Explicit(1, 2, 3, 4, 5, 6);
        assert_eq!(padding.as_tuple(), (1, 2, 3, 4, 5, 6));
    }

    #[test]
    #[should_panic(expected = "as_tuple() only works with Explicit padding")]
    fn test_padding_config_3d_as_tuple_valid_panics() {
        let padding = PaddingConfig3d::Valid;
        let _ = padding.as_tuple();
    }

    #[test]
    #[should_panic(expected = "as_tuple() only works with Explicit padding")]
    fn test_padding_config_3d_as_tuple_same_panics() {
        let padding = PaddingConfig3d::Same;
        let _ = padding.as_tuple();
    }

    #[test]
    fn test_padding_config_3d_calculate_valid() {
        let padding = PaddingConfig3d::Valid;
        assert_eq!(
            padding.calculate_padding_3d(10, 10, 10, &[3, 3, 3], &[1, 1, 1]),
            [0, 0, 0]
        );
    }

    #[test]
    fn test_padding_config_3d_calculate_explicit_symmetric() {
        let padding = PaddingConfig3d::Explicit(1, 2, 3, 1, 2, 3);
        assert_eq!(
            padding.calculate_padding_3d(10, 10, 10, &[3, 3, 3], &[1, 1, 1]),
            [1, 2, 3]
        );
    }

    #[test]
    #[should_panic(
        expected = "Asymmetric padding should be handled via calculate_padding_3d_pairs"
    )]
    fn test_padding_config_3d_calculate_explicit_asymmetric_panics() {
        let padding = PaddingConfig3d::Explicit(1, 2, 3, 4, 5, 6);
        let _ = padding.calculate_padding_3d(10, 10, 10, &[3, 3, 3], &[1, 1, 1]);
    }
}
