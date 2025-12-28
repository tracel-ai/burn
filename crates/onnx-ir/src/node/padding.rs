//! # Padding Configuration Utilities
//!
//! Padding configuration types for 1D, 2D, and 3D operations.
//!
//! Provides `PaddingConfig1d`, `PaddingConfig2d`, `PaddingConfig3d` enums and helper
//! functions to convert ONNX padding arrays.

use std::fmt;

/// Padding configuration for 1D operations such as convolution
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum PaddingConfig1d {
    /// No padding (valid padding)
    #[default]
    Valid,
    /// Explicit padding with values for left and right sides
    /// Format: (left, right)
    /// For symmetric padding, use the same value for both (e.g., `Explicit(1, 1)`).
    Explicit(usize, usize),
}

impl fmt::Display for PaddingConfig1d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PaddingConfig1d::Valid => write!(f, "Valid"),
            PaddingConfig1d::Explicit(left, right) => write!(f, "Explicit({left}, {right})"),
        }
    }
}

impl PaddingConfig1d {
    /// Returns true if this padding configuration is asymmetric (left != right)
    pub fn is_asymmetric(&self) -> bool {
        match self {
            PaddingConfig1d::Explicit(left, right) => left != right,
            _ => false,
        }
    }

    /// Returns the padding values as (left, right) tuple
    pub fn as_tuple(&self) -> (usize, usize) {
        match self {
            PaddingConfig1d::Valid => (0, 0),
            PaddingConfig1d::Explicit(left, right) => (*left, *right),
        }
    }
}

/// Calculate the padding configuration for a 1D operations such as Convolution and Pooling.
///
/// # Arguments
///
/// * `pads` - The padding values [left, right]
///
/// # Panics
///
/// * If the padding is negative
///
/// # Returns
///
/// * The padding configuration (Valid or Explicit)
///
/// # Remarks
///
/// This function is used when the padding is specified as a list of integers,
/// and not used when the padding is specified as a string, e.g. "SAME_UPPER".
pub(crate) fn padding_config_1d(pads: &[i64]) -> PaddingConfig1d {
    let [left, right] = [pads[0], pads[1]];

    if left < 0 || right < 0 {
        panic!("Negative pad values are not supported");
    } else if left == 0 && right == 0 {
        PaddingConfig1d::Valid
    } else {
        PaddingConfig1d::Explicit(left as usize, right as usize)
    }
}

/// Padding configuration for 2D operations such as convolution
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum PaddingConfig2d {
    /// No padding (valid padding)
    #[default]
    Valid,
    /// Explicit padding with values for each side
    /// Format: (top, left, bottom, right)
    /// For symmetric padding, use matching values (e.g., `Explicit(1, 1, 1, 1)`).
    Explicit(usize, usize, usize, usize),
}

impl fmt::Display for PaddingConfig2d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PaddingConfig2d::Valid => write!(f, "Valid"),
            PaddingConfig2d::Explicit(top, left, bottom, right) => {
                write!(f, "Explicit({top}, {left}, {bottom}, {right})")
            }
        }
    }
}

impl PaddingConfig2d {
    /// Returns true if this padding configuration is asymmetric (top != bottom or left != right)
    pub fn is_asymmetric(&self) -> bool {
        match self {
            PaddingConfig2d::Explicit(top, left, bottom, right) => top != bottom || left != right,
            _ => false,
        }
    }

    /// Returns the padding values as (top, left, bottom, right) tuple
    pub fn as_tuple(&self) -> (usize, usize, usize, usize) {
        match self {
            PaddingConfig2d::Valid => (0, 0, 0, 0),
            PaddingConfig2d::Explicit(top, left, bottom, right) => (*top, *left, *bottom, *right),
        }
    }
}

/// Calculate the padding configuration for a 2D operations such as Convolution and Pooling.
///
/// # Arguments
///
/// * `pads` - The padding values [top, left, bottom, right] (ONNX format)
///
/// # Panics
///
/// * If the padding is negative
///
/// # Returns
///
/// * The padding configuration (Valid or Explicit)
///
/// # Remarks
///
/// This function is used when the padding is specified as a list of integers,
/// and not used when the padding is specified as a string, e.g. "SAME_UPPER".
pub(crate) fn padding_config_2d(pads: &[i64]) -> PaddingConfig2d {
    let [top, left, bottom, right] = [pads[0], pads[1], pads[2], pads[3]];

    if left < 0 || right < 0 || top < 0 || bottom < 0 {
        panic!("Negative pad values are not supported");
    } else if left == 0 && right == 0 && top == 0 && bottom == 0 {
        PaddingConfig2d::Valid
    } else {
        PaddingConfig2d::Explicit(top as usize, left as usize, bottom as usize, right as usize)
    }
}

/// Padding configuration for 3D operations such as convolution
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum PaddingConfig3d {
    /// No padding (valid padding)
    #[default]
    Valid,
    /// Explicit padding with values for each side
    /// Format: (front, top, left, back, bottom, right)
    /// For symmetric padding, use matching values (e.g., `Explicit(1, 1, 1, 1, 1, 1)`).
    Explicit(usize, usize, usize, usize, usize, usize),
}

impl fmt::Display for PaddingConfig3d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PaddingConfig3d::Valid => write!(f, "Valid"),
            PaddingConfig3d::Explicit(front, top, left, back, bottom, right) => {
                write!(
                    f,
                    "Explicit({front}, {top}, {left}, {back}, {bottom}, {right})"
                )
            }
        }
    }
}

impl PaddingConfig3d {
    /// Returns true if this padding configuration is asymmetric
    pub fn is_asymmetric(&self) -> bool {
        match self {
            PaddingConfig3d::Explicit(front, top, left, back, bottom, right) => {
                front != back || top != bottom || left != right
            }
            _ => false,
        }
    }

    /// Returns the padding values as (front, top, left, back, bottom, right) tuple
    pub fn as_tuple(&self) -> (usize, usize, usize, usize, usize, usize) {
        match self {
            PaddingConfig3d::Valid => (0, 0, 0, 0, 0, 0),
            PaddingConfig3d::Explicit(front, top, left, back, bottom, right) => {
                (*front, *top, *left, *back, *bottom, *right)
            }
        }
    }
}

/// Calculate the padding configuration for a 3D operations such as Convolution and Pooling.
///
/// # Arguments
///
/// * `pads` - The padding values [front, top, left, back, bottom, right] (ONNX format)
///
/// # Panics
///
/// * If the padding is negative
///
/// # Returns
///
/// * The padding configuration (Valid or Explicit)
///
/// # Remarks
///
/// This function is used when the padding is specified as a list of integers,
/// and not used when the padding is specified as a string, e.g. "SAME_UPPER".
pub(crate) fn padding_config_3d(pads: &[i64]) -> PaddingConfig3d {
    let [front, top, left, back, bottom, right] =
        [pads[0], pads[1], pads[2], pads[3], pads[4], pads[5]];

    if left < 0 || right < 0 || top < 0 || bottom < 0 || front < 0 || back < 0 {
        panic!("Negative pad values are not supported");
    } else if left == 0 && right == 0 && top == 0 && bottom == 0 && front == 0 && back == 0 {
        PaddingConfig3d::Valid
    } else {
        PaddingConfig3d::Explicit(
            front as usize,
            top as usize,
            left as usize,
            back as usize,
            bottom as usize,
            right as usize,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 1D padding tests
    #[test]
    fn test_padding_config_1d_valid() {
        let pads = vec![0, 0];
        let config = padding_config_1d(&pads);
        assert!(matches!(config, PaddingConfig1d::Valid));
    }

    #[test]
    fn test_padding_config_1d_explicit_symmetric() {
        let pads = vec![2, 2];
        let config = padding_config_1d(&pads);
        assert!(matches!(config, PaddingConfig1d::Explicit(2, 2)));
        assert!(!config.is_asymmetric());
        assert_eq!(config.as_tuple(), (2, 2));
    }

    #[test]
    fn test_padding_config_1d_explicit_asymmetric() {
        let pads = vec![1, 2];
        let config = padding_config_1d(&pads);
        assert!(matches!(config, PaddingConfig1d::Explicit(1, 2)));
        assert!(config.is_asymmetric());
        assert_eq!(config.as_tuple(), (1, 2));
    }

    #[test]
    #[should_panic(expected = "Negative pad values are not supported")]
    fn test_padding_config_1d_negative() {
        let pads = vec![-1, -1];
        let _ = padding_config_1d(&pads);
    }

    // 2D padding tests
    #[test]
    fn test_padding_config_2d_valid() {
        let pads = vec![0, 0, 0, 0];
        let config = padding_config_2d(&pads);
        assert!(matches!(config, PaddingConfig2d::Valid));
        assert!(!config.is_asymmetric());
    }

    #[test]
    fn test_padding_config_2d_explicit_symmetric() {
        let pads = vec![2, 2, 2, 2];
        let config = padding_config_2d(&pads);
        assert!(matches!(config, PaddingConfig2d::Explicit(2, 2, 2, 2)));
        assert!(!config.is_asymmetric());
        assert_eq!(config.as_tuple(), (2, 2, 2, 2));
    }

    #[test]
    fn test_padding_config_2d_explicit_asymmetric() {
        // pads = [top, left, bottom, right]
        let pads = vec![1, 2, 3, 4];
        let config = padding_config_2d(&pads);
        assert!(matches!(config, PaddingConfig2d::Explicit(1, 2, 3, 4)));
        assert!(config.is_asymmetric());
        assert_eq!(config.as_tuple(), (1, 2, 3, 4));
    }

    #[test]
    fn test_padding_config_2d_explicit_asymmetric_top_bottom() {
        // top != bottom but left == right
        let pads = vec![1, 2, 3, 2];
        let config = padding_config_2d(&pads);
        assert!(matches!(config, PaddingConfig2d::Explicit(1, 2, 3, 2)));
        assert!(config.is_asymmetric());
    }

    #[test]
    fn test_padding_config_2d_explicit_asymmetric_left_right() {
        // left != right but top == bottom
        let pads = vec![2, 1, 2, 3];
        let config = padding_config_2d(&pads);
        assert!(matches!(config, PaddingConfig2d::Explicit(2, 1, 2, 3)));
        assert!(config.is_asymmetric());
    }

    #[test]
    #[should_panic(expected = "Negative pad values are not supported")]
    fn test_padding_config_2d_negative() {
        let pads = vec![-1, -1, -1, -1];
        let _ = padding_config_2d(&pads);
    }

    // 3D padding tests
    #[test]
    fn test_padding_config_3d_valid() {
        let pads = vec![0, 0, 0, 0, 0, 0];
        let config = padding_config_3d(&pads);
        assert!(matches!(config, PaddingConfig3d::Valid));
        assert!(!config.is_asymmetric());
    }

    #[test]
    fn test_padding_config_3d_explicit_symmetric() {
        let pads = vec![2, 3, 1, 2, 3, 1];
        let config = padding_config_3d(&pads);
        assert!(matches!(
            config,
            PaddingConfig3d::Explicit(2, 3, 1, 2, 3, 1)
        ));
        assert!(!config.is_asymmetric());
        assert_eq!(config.as_tuple(), (2, 3, 1, 2, 3, 1));
    }

    #[test]
    fn test_padding_config_3d_explicit_asymmetric() {
        // pads = [front, top, left, back, bottom, right]
        let pads = vec![1, 2, 3, 4, 5, 6];
        let config = padding_config_3d(&pads);
        assert!(matches!(
            config,
            PaddingConfig3d::Explicit(1, 2, 3, 4, 5, 6)
        ));
        assert!(config.is_asymmetric());
        assert_eq!(config.as_tuple(), (1, 2, 3, 4, 5, 6));
    }

    #[test]
    fn test_padding_config_3d_explicit_asymmetric_partial() {
        // Only front != back
        let pads = vec![1, 3, 1, 2, 3, 1];
        let config = padding_config_3d(&pads);
        assert!(matches!(
            config,
            PaddingConfig3d::Explicit(1, 3, 1, 2, 3, 1)
        ));
        assert!(config.is_asymmetric());
    }

    #[test]
    #[should_panic(expected = "Negative pad values are not supported")]
    fn test_padding_config_3d_negative() {
        let pads = vec![-1, -1, -1, -1, -1, -1];
        let _ = padding_config_3d(&pads);
    }
}
