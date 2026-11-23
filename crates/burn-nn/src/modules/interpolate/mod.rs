mod interpolate1d;
mod interpolate2d;

pub use interpolate1d::*;
pub use interpolate2d::*;

use burn_core as burn;

use burn::config::Config;
use burn::tensor::ops::{CoordinateTransformMode, InterpolateMode as OpsInterpolateMode};

/// Algorithm used for downsampling and upsampling
///
/// This enum defines different interpolation modes for resampling data.
#[derive(Config, Debug)]
pub enum InterpolateMode {
    /// Nearest-neighbor floor interpolation
    ///
    /// This mode selects the value with floor mapping to a sample point for each output pixel.
    /// It is applicable for both temporal and spatial data.
    Nearest,

    /// Nearest-neighbor exact interpolation
    ///
    /// This mode selects the value of the nearest sample point for each output pixel.
    /// It is applicable for both temporal and spatial data.
    NearestExact,

    /// Linear interpolation
    ///
    /// This mode calculates the output value using linear
    /// interpolation between nearby sample points.
    ///
    /// It is applicable for both temporal and spatial data.
    Linear {
        /// Mapping from output coordinates to input coordinates.
        coordinate_transform: CoordinateTransformMode,
    },

    /// Cubic interpolation
    ///
    /// This mode uses cubic interpolation to calculate the output value
    /// based on surrounding sample points.
    ///
    /// It is applicable for both temporal and spatial data and generally
    /// provides smoother results than linear interpolation.
    Cubic {
        /// Mapping from output coordinates to input coordinates.
        coordinate_transform: CoordinateTransformMode,
        /// Cubic convolution coefficient, conventionally `-0.75`.
        a: f32,
    },

    /// Lanczos3 interpolation
    ///
    /// This mode uses a 6-tap sinc-based Lanczos filter (a=3) to calculate
    /// the output value. It generally provides high-quality results,
    /// especially for downsampling.
    Lanczos {
        /// Mapping from output coordinates to input coordinates.
        coordinate_transform: CoordinateTransformMode,
    },
}

impl From<InterpolateMode> for OpsInterpolateMode {
    fn from(mode: InterpolateMode) -> Self {
        match mode {
            InterpolateMode::Nearest => OpsInterpolateMode::Nearest,
            InterpolateMode::NearestExact => OpsInterpolateMode::NearestExact,
            InterpolateMode::Linear {
                coordinate_transform,
            } => OpsInterpolateMode::Bilinear {
                coordinate_transform,
            },
            InterpolateMode::Cubic {
                coordinate_transform,
                a,
            } => OpsInterpolateMode::Bicubic {
                coordinate_transform,
                a,
            },
            InterpolateMode::Lanczos {
                coordinate_transform,
            } => OpsInterpolateMode::Lanczos3 {
                coordinate_transform,
            },
        }
    }
}
