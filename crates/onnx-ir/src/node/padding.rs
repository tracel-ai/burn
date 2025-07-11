use std::fmt;

/// Padding configuration for 1D operations such as convolution
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PaddingConfig1d {
    /// No padding (valid padding)
    Valid,
    /// Explicit padding with a specific size
    Explicit(usize),
}

impl fmt::Display for PaddingConfig1d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PaddingConfig1d::Valid => write!(f, "Valid"),
            PaddingConfig1d::Explicit(size) => write!(f, "Explicit({size})"),
        }
    }
}

/// Calculate the padding configuration for a 1D operations such as Convolution and Pooling.
///
/// # Arguments
///
/// * `pads` - The padding values
///
/// # Panics
///
/// * If the padding is negative
/// * If the padding is not symmetric
///
/// # Returns
///
/// * The padding configuration
///
/// # Remarks
///
/// This function is used when the padding is specified as a list of integers,
/// and not used when the padding is specified as a string, e.g. "SAME_UPPER".
pub fn padding_config_1d(pads: &[i64]) -> PaddingConfig1d {
    let [left, right] = [pads[0], pads[1]];

    if left < 0 || right < 0 {
        panic!("Negative pad values are not supported");
    } else if left != right {
        panic!("Asymmetric padding is not supported");
    } else if left == 0 && right == 0 {
        // i.e. [0, 0]
        PaddingConfig1d::Valid
    } else if left == right {
        // i.e. [2, 2]
        PaddingConfig1d::Explicit(left as usize)
    } else {
        // Unaccounted for padding configuration
        panic!("Padding configuration ({pads:?}) not supported");
    }
}

/// Padding configuration for 2D operations such as convolution
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PaddingConfig2d {
    /// No padding (valid padding)
    Valid,
    /// Explicit padding with specific width and height
    Explicit(usize, usize),
}

impl fmt::Display for PaddingConfig2d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PaddingConfig2d::Valid => write!(f, "Valid"),
            PaddingConfig2d::Explicit(width, height) => {
                write!(f, "Explicit({width}, {height})")
            }
        }
    }
}

/// Padding configuration for 3D operations such as convolution
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PaddingConfig3d {
    /// No padding (valid padding)
    Valid,
    /// Explicit padding with specific width, height, and depth
    Explicit(usize, usize, usize),
}

impl fmt::Display for PaddingConfig3d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PaddingConfig3d::Valid => write!(f, "Valid"),
            PaddingConfig3d::Explicit(width, height, depth) => {
                write!(f, "Explicit({width}, {height}, {depth})")
            }
        }
    }
}

/// Calculate the padding configuration for a 2D operations such as Convolution and Pooling.
///
/// # Arguments
///
/// * `pads` - The padding values [left, right, top, bottom]
///
/// # Panics
///
/// * If the padding is negative
/// * If the padding is not symmetric
///
/// # Returns
///
/// * The padding configuration
///
/// # Remarks
///
/// This function is used when the padding is specified as a list of integers,
/// and not used when the padding is specified as a string, e.g. "SAME_UPPER".
pub fn padding_config_2d(pads: &[i64]) -> PaddingConfig2d {
    let [top, left, bottom, right] = [pads[0], pads[1], pads[2], pads[3]];

    if left < 0 || right < 0 || top < 0 || bottom < 0 {
        panic!("Negative pad values are not supported");
    } else if left != right || top != bottom {
        panic!("Asymmetric padding is not supported");
    } else if left == 0 && right == 0 && top == 0 && bottom == 0 {
        PaddingConfig2d::Valid
    } else if left == right && top == bottom {
        PaddingConfig2d::Explicit(top as usize, left as usize)
    } else {
        // Unaccounted for padding configuration
        panic!("Padding configuration ({pads:?}) not supported");
    }
}

/// Calculate the padding configuration for a 3D operations such as Convolution and Pooling.
///
/// # Arguments
///
/// * `pads` - The padding values [left, right, top, bottom, front, back]
///
/// # Panics
///
/// * If the padding is negative
/// * If the padding is not symmetric
///
/// # Returns
///
/// * The padding configuration
///
/// # Remarks
///
/// This function is used when the padding is specified as a list of integers,
/// and not used when the padding is specified as a string, e.g. "SAME_UPPER".
pub fn padding_config_3d(pads: &[i64]) -> PaddingConfig3d {
    let [front, top, left, back, bottom, right] =
        [pads[0], pads[1], pads[2], pads[3], pads[4], pads[5]];

    if left < 0 || right < 0 || top < 0 || bottom < 0 || front < 0 || back < 0 {
        panic!("Negative pad values are not supported");
    } else if left != right || top != bottom || front != back {
        panic!("Asymmetric padding is not supported");
    } else if left == 0 && right == 0 && top == 0 && bottom == 0 && front == 0 && back == 0 {
        PaddingConfig3d::Valid
    } else if left == right && top == bottom && front == back {
        PaddingConfig3d::Explicit(front as usize, top as usize, left as usize)
    } else {
        // Unaccounted for padding configuration
        panic!("Padding configuration ({pads:?}) not supported");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_padding_config_2d_valid() {
        let pads = vec![0, 0, 0, 0];
        let config = padding_config_2d(&pads);
        assert!(matches!(config, PaddingConfig2d::Valid));
    }

    #[test]
    fn test_padding_config_2d_explicit() {
        let pads = vec![2, 2, 2, 2];
        let config = padding_config_2d(&pads);
        assert!(matches!(config, PaddingConfig2d::Explicit(2, 2)));
    }

    #[test]
    #[should_panic(expected = "Asymmetric padding is not supported")]
    fn test_padding_config_2d_asymmetric() {
        let pads = vec![2, 3, 2, 2];
        let _ = padding_config_2d(&pads);
    }

    #[test]
    #[should_panic(expected = "Negative pad values are not supported")]
    fn test_padding_config_2d_negative() {
        let pads = vec![-1, -1, -1, -1];
        let _ = padding_config_2d(&pads);
    }

    #[test]
    fn test_padding_config_3d_valid() {
        let pads = vec![0, 0, 0, 0, 0, 0];
        let config = padding_config_3d(&pads);
        assert!(matches!(config, PaddingConfig3d::Valid));
    }

    #[test]
    fn test_padding_config_3d_explicit() {
        let pads = vec![2, 3, 1, 2, 3, 1];
        let config = padding_config_3d(&pads);
        assert!(matches!(config, PaddingConfig3d::Explicit(2, 3, 1)));
    }

    #[test]
    #[should_panic(expected = "Asymmetric padding is not supported")]
    fn test_padding_config_3d_asymmetric() {
        let pads = vec![2, 3, 1, 3, 3, 1];
        let _ = padding_config_3d(&pads);
    }

    #[test]
    #[should_panic(expected = "Negative pad values are not supported")]
    fn test_padding_config_3d_negative() {
        let pads = vec![-1, -1, -1, -1, -1, -1];
        let _ = padding_config_3d(&pads);
    }
}
