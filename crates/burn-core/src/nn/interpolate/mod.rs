mod interpolate1d;
mod interpolate2d;

pub use interpolate1d::*;
pub use interpolate2d::*;

/// Coordinate transformation mode using scale_factor
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum CoordinateTransformationMode {
    /// x_resized = (x_original + 0.5) * scale - 0.5
    HalfPixel,

    /// x_resized = x_original * scale
    Asymmetric,
}
