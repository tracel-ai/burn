use burn_tensor::Element;
use serde::{de::DeserializeOwned, Serialize};

pub trait RecordSettings: Send + Sync + core::fmt::Debug + core::default::Default + Clone {
    type FloatElem: Element + Serialize + DeserializeOwned;
    type IntElem: Element + Serialize + DeserializeOwned;
}

/// Default record settings.
#[derive(Debug, Default, Clone)]
pub struct FullPrecisionSettings;

/// Record settings optimized for compactness.
#[cfg(feature = "std")]
#[derive(Debug, Default, Clone)]
pub struct HalfPrecisionSettings;

/// Record settings optimized for precision.
#[cfg(feature = "std")]
#[derive(Debug, Default, Clone)]
pub struct DoublePrecisionSettings;

impl RecordSettings for FullPrecisionSettings {
    type FloatElem = f32;
    type IntElem = f32;
}

impl RecordSettings for DoublePrecisionSettings {
    type FloatElem = f64;
    type IntElem = i64;
}

#[cfg(feature = "std")]
impl RecordSettings for HalfPrecisionSettings {
    type FloatElem = half::f16;
    type IntElem = i16;
}
