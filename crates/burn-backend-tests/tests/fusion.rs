//! Burn tensor and autodiff tests for CubeCL backends with fusion enabled.

pub type FloatElemType = f32;
pub type IntElemType = i32;

#[cfg(feature = "cube")]
#[path = "common/fusion.rs"]
mod fusion;
