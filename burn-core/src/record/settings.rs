use super::Recorder;
use burn_tensor::Element;
use serde::{de::DeserializeOwned, Serialize};

pub trait RecordSettings: Send + Sync + core::fmt::Debug + core::default::Default {
    type FloatElem: Element + Serialize + DeserializeOwned;
    type IntElem: Element + Serialize + DeserializeOwned;
    type Recorder: Recorder;
}

/// Default record settings.
#[derive(Debug, Default)]
pub struct DefaultRecordSettings;
/// Training settings compatible with no-std inference.
#[derive(Debug, Default)]
pub struct NoStdTrainingRecordSettings;
/// Inference settings compatible with no-std.
#[derive(Debug, Default)]
pub struct NoStdInferenceRecordSettings;
/// Debug record settings.
///
/// # Notes
///
/// The recorder used in this settings is human readable.
#[derive(Debug, Default)]
pub struct DebugRecordSettings;

impl RecordSettings for DefaultRecordSettings {
    #[cfg(feature = "std")]
    type FloatElem = half::f16;
    #[cfg(not(feature = "std"))]
    type FloatElem = f32;
    type IntElem = i16;
    #[cfg(feature = "std")]
    type Recorder = crate::record::FileBinGzRecorder;
    #[cfg(not(feature = "std"))]
    type Recorder = crate::record::InMemoryBinRecorder;
}

#[cfg(feature = "std")]
impl RecordSettings for NoStdTrainingRecordSettings {
    type FloatElem = f32;
    type IntElem = i32;
    type Recorder = crate::record::FileBinRecorder;
}

impl RecordSettings for NoStdInferenceRecordSettings {
    type FloatElem = f32;
    type IntElem = i32;
    type Recorder = crate::record::InMemoryBinRecorder;
}

#[cfg(feature = "std")]
impl RecordSettings for DebugRecordSettings {
    type FloatElem = f32;
    type IntElem = i32;
    type Recorder = crate::record::FilePrettyJsonRecorder;
}
