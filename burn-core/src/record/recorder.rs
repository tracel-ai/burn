use alloc::format;
use alloc::string::String;
use serde::{Deserialize, Serialize};

use super::{
    BinBytesRecorder, BinFileRecorder, BinGzFileRecorder, DefaultFileRecorder,
    FullPrecisionSettings, HalfPrecisionSettings, PrecisionSettings, PrettyJsonFileRecorder,
    Record,
};

/// Record any item implementing [Serialize](Serialize) and [DeserializeOwned](DeserializeOwned).
pub trait Recorder: Send + Sync + core::default::Default + core::fmt::Debug + Clone {
    type Settings: PrecisionSettings;
    /// Arguments used to record objects.
    type RecordArgs: Clone;
    /// Record output type.
    type RecordOutput;
    /// Arguments used to load recorded objects.
    type LoadArgs: Clone;

    /// Record using the given [settings](RecordSettings).
    fn record<R: Record>(
        &self,
        record: R,
        args: Self::RecordArgs,
    ) -> Result<Self::RecordOutput, RecorderError> {
        let metadata = BurnMetadata::new(
            core::any::type_name::<<Self::Settings as PrecisionSettings>::FloatElem>().to_string(),
            core::any::type_name::<<Self::Settings as PrecisionSettings>::IntElem>().to_string(),
            core::any::type_name::<Self>().to_string(),
            env!("CARGO_PKG_VERSION").to_string(),
            format!("{:?}", Self::Settings::default()),
        );
        let item = record.into_item::<Self::Settings>();
        let item = BurnRecord::new(metadata, item);

        self.save_item::<R>(item, args)
    }

    /// Load an item from the given arguments.
    fn load<R: Record>(&self, args: Self::LoadArgs) -> Result<R, RecorderError> {
        let item = self.load_item::<R>(args.clone())?;

        Ok(R::from_item(item.item))
    }

    fn save_item<R: Record>(
        &self,
        item: BurnRecord<R::Item<Self::Settings>>,
        args: Self::RecordArgs,
    ) -> Result<Self::RecordOutput, RecorderError>;
    fn load_item<R: Record>(
        &self,
        args: Self::LoadArgs,
    ) -> Result<BurnRecord<R::Item<Self::Settings>>, RecorderError>;
}

#[derive(Debug)]
pub enum RecorderError {
    FileNotFound(String),
    Unknown(String),
}

impl core::fmt::Display for RecorderError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(format!("{self:?}").as_str())
    }
}

// TODO: Move from std to core after Error is core (see https://github.com/rust-lang/rust/issues/103765)
#[cfg(feature = "std")]
impl std::error::Error for RecorderError {}

pub(crate) fn bin_config() -> bincode::config::Configuration {
    bincode::config::standard()
}

#[derive(new, Debug, Serialize, Deserialize)]
pub struct BurnMetadata {
    float: String,
    int: String,
    format: String,
    version: String,
    settings: String,
}

#[derive(new, Serialize, Deserialize)]
pub struct BurnRecord<I> {
    pub metadata: BurnMetadata,
    pub item: I,
}

#[derive(new, Serialize, Deserialize)]
struct BurnRecordNoItem {
    metadata: BurnMetadata,
}

/// Default recorder.
///
/// It uses the [named msgpack](rmp_serde) format for serialization with full precision.
pub type DefaultRecorder = DefaultFileRecorder<FullPrecisionSettings>;

/// Recorder optimized for compactness.
///
/// It uses the [named msgpack](rmp_serde) format for serialization with half precision.
/// If you are looking for the recorder that offers the smallest file size, have a look at
/// [sensitive compact recorder](SensitiveCompactRecorder).
pub type CompactRecorder = DefaultFileRecorder<HalfPrecisionSettings>;

/// Recorder optimized for compactness making it a good choice for model deployment.
///
/// It uses the [bincode](bincode) format for serialization and half precision.
/// This format is not resilient to type changes since no metadata is encoded.
/// Favor [default recorder](DefaultRecorder) or [compact recorder](CompactRecorder)
/// for long term data storage.
pub type SensitiveCompactRecorder = BinGzFileRecorder<HalfPrecisionSettings>;

/// Training recorder compatible with no-std inference.
pub type NoStdTrainingRecorder = BinFileRecorder<FullPrecisionSettings>;

/// Inference recorder compatible with no-std.
pub type NoStdInferenceRecorder = BinBytesRecorder<FullPrecisionSettings>;

/// Debug recorder.
///
/// It uses the [pretty json](serde_json) format for serialization with full precision making it
/// human readable.
pub type DebugRecordSettings = PrettyJsonFileRecorder<FullPrecisionSettings>;

#[cfg(all(test, feature = "std"))]
mod tests {
    static FILE_PATH: &str = "/tmp/burn_test_record";

    use core::marker::PhantomData;

    use super::*;
    use crate::record::JsonGzFileRecorder;
    use burn_tensor::{Element, ElementConversion};
    use serde::de::DeserializeOwned;

    #[test]
    #[should_panic]
    fn err_when_invalid_object() {
        #[derive(Debug, Default)]
        pub struct TestSettings<F> {
            float: PhantomData<F>,
        }

        #[derive(new, Serialize, Deserialize)]
        struct Item<S: PrecisionSettings> {
            value: S::FloatElem,
        }

        impl<D: PrecisionSettings> Record for Item<D> {
            type Item<S: PrecisionSettings> = Item<S>;

            fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
                Item {
                    value: self.value.elem(),
                }
            }

            fn from_item<S: PrecisionSettings>(item: Self::Item<S>) -> Self {
                Item {
                    value: item.value.elem(),
                }
            }
        }

        let item = Item::<TestSettings<f32>>::new(16.elem());

        // Serialize in f32.
        item.record::<TestSettings<f32>>(FILE_PATH.into()).unwrap();
        // Can't deserialize u8 into f32.
        Item::<TestSettings<f32>>::load::<TestSettings<u8>>(FILE_PATH.into()).unwrap();
    }
}
