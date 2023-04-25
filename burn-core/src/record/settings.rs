use super::Recorder;
use burn_tensor::Element;
use serde::{de::DeserializeOwned, Serialize};

pub trait RecordSettings: Send + Sync + core::fmt::Debug + core::default::Default {
    type FloatElem: Element + Serialize + DeserializeOwned;
    type IntElem: Element + Serialize + DeserializeOwned;
    type Recorder: Recorder;
}

/// Default record settings.
///
/// It uses the [named msgpack](rmp_serde) format for serialization with full precision to encode
/// numbers.
#[cfg(feature = "std")]
#[derive(Debug, Default)]
pub struct DefaultRecordSettings;

/// Record settings optimized for compactness.
///
/// It uses the [named msgpack](rmp_serde) format for serialization with half precision to encode numbers.
/// If you are looking for the settings that offers the smallest file size, have a look at
/// [sensitive compact settings](SentitiveCompactRecordSettings).
#[cfg(feature = "std")]
#[derive(Debug, Default)]
pub struct CompactRecordSettings;

/// Record settings optimized for compactness making it a good choice for model deployment.
///
/// It uses the [bincode](bincode) format for serialization and half precision to encode numbers.
/// This format is not resilient to type changes since no metadata is encoded.
/// Favor [default settings](DebugRecordSettings) or [compact settings](CompactRecordSettings)
/// for long term data storage.
#[cfg(feature = "std")]
#[derive(Debug, Default)]
pub struct SentitiveCompactRecordSettings;

/// Training settings compatible with no-std inference.
#[cfg(feature = "std")]
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

#[cfg(feature = "std")]
impl RecordSettings for DefaultRecordSettings {
    type FloatElem = f32;
    type IntElem = f32;
    type Recorder = crate::record::FileNamedMpkGzRecorder;
}

#[cfg(feature = "std")]
impl RecordSettings for CompactRecordSettings {
    type FloatElem = half::f16;
    type IntElem = i16;
    type Recorder = crate::record::FileNamedMpkGzRecorder;
}

#[cfg(feature = "std")]
impl RecordSettings for SentitiveCompactRecordSettings {
    type FloatElem = half::f16;
    type IntElem = i16;
    type Recorder = crate::record::FileBinGzRecorder;
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
