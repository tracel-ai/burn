use core::any::type_name;
use core::marker::PhantomData;

use alloc::format;
use alloc::string::{String, ToString};
use burn_tensor::backend::Backend;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use super::{BinBytesRecorder, FullPrecisionSettings, PrecisionSettings, Record};

#[cfg(feature = "std")]
use super::{
    BinFileRecorder, BinGzFileRecorder, DefaultFileRecorder, HalfPrecisionSettings,
    PrettyJsonFileRecorder,
};

/// Record any item implementing [Serialize](Serialize) and [DeserializeOwned](DeserializeOwned).
pub trait Recorder<B: Backend>:
    Send + Sync + core::default::Default + core::fmt::Debug + Clone
{
    /// Type of the settings used by the recorder.
    type Settings: PrecisionSettings;

    /// Arguments used to record objects.
    type RecordArgs: Clone;

    /// Record output type.
    type RecordOutput;

    /// Arguments used to load recorded objects.
    type LoadArgs;

    /// Records an item.
    ///
    /// # Arguments
    ///
    /// * `record` - The item to record.
    /// * `args` - Arguments used to record the item.
    ///
    /// # Returns
    ///
    /// The output of the recording.
    fn record<R>(
        &self,
        record: R,
        args: Self::RecordArgs,
    ) -> Result<Self::RecordOutput, RecorderError>
    where
        R: Record<B>,
    {
        let item = record.into_item::<Self::Settings>();
        let item = BurnRecord::new::<Self>(item);

        self.save_item(item, args)
    }

    /// Load an item from the given arguments.
    fn load<R>(&self, mut args: Self::LoadArgs, device: &B::Device) -> Result<R, RecorderError>
    where
        R: Record<B>,
    {
        let item: BurnRecord<R::Item<Self::Settings>, B> =
            self.load_item(&mut args).map_err(|err| {
                if let Ok(record) = self.load_item::<BurnRecordNoItem>(&mut args) {
                    let mut message = "Unable to load record.".to_string();
                    let metadata = recorder_metadata::<Self, B>();
                    if metadata.float != record.metadata.float {
                        message += format!(
                            "\nMetadata has a different float type: Actual {:?}, Expected {:?}",
                            record.metadata.float, metadata.float
                        )
                        .as_str();
                    }
                    if metadata.int != record.metadata.int {
                        message += format!(
                            "\nMetadata has a different int type: Actual {:?}, Expected {:?}",
                            record.metadata.int, metadata.int
                        )
                        .as_str();
                    }
                    if metadata.format != record.metadata.format {
                        message += format!(
                            "\nMetadata has a different format: Actual {:?}, Expected {:?}",
                            record.metadata.format, metadata.format
                        )
                        .as_str();
                    }
                    if metadata.version != record.metadata.version {
                        message += format!(
                            "\nMetadata has a different Burn version: Actual {:?}, Expected {:?}",
                            record.metadata.version, metadata.version
                        )
                        .as_str();
                    }

                    message += format!("\nError: {err:?}").as_str();

                    return RecorderError::Unknown(message);
                }

                err
            })?;

        Ok(R::from_item(item.item, device))
    }

    /// Saves an item.
    ///
    /// This method is used by [record](Recorder::record) to save the item.
    ///
    /// # Arguments
    ///
    /// * `item` - Item to save.
    /// * `args` - Arguments to use to save the item.
    ///
    /// # Returns
    ///
    /// The output of the save operation.
    fn save_item<I: Serialize>(
        &self,
        item: I,
        args: Self::RecordArgs,
    ) -> Result<Self::RecordOutput, RecorderError>;

    /// Loads an item.
    ///
    /// This method is used by [load](Recorder::load) to load the item.
    ///
    /// # Arguments
    ///
    /// * `args` - Arguments to use to load the item.
    ///
    /// # Returns
    ///
    /// The loaded item.
    fn load_item<I>(&self, args: &mut Self::LoadArgs) -> Result<I, RecorderError>
    where
        I: DeserializeOwned;
}

fn recorder_metadata<R, B>() -> BurnMetadata
where
    R: Recorder<B>,
    B: Backend,
{
    BurnMetadata::new(
        type_name::<<R::Settings as PrecisionSettings>::FloatElem>().to_string(),
        type_name::<<R::Settings as PrecisionSettings>::IntElem>().to_string(),
        type_name::<R>().to_string(),
        env!("CARGO_PKG_VERSION").to_string(),
        format!("{:?}", R::Settings::default()),
    )
}

/// Error that can occur when using a [Recorder](Recorder).
#[derive(Debug)]
pub enum RecorderError {
    /// File not found.
    FileNotFound(String),

    /// Failed to read file.
    DeserializeError(String),

    /// Other error.
    Unknown(String),
}

impl core::fmt::Display for RecorderError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(format!("{self:?}").as_str())
    }
}

impl core::error::Error for RecorderError {}

pub(crate) fn bin_config() -> bincode::config::Configuration {
    bincode::config::standard()
}

/// Metadata of a record.
#[derive(new, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct BurnMetadata {
    /// Float type used to record the item.
    pub float: String,

    /// Int type used to record the item.
    pub int: String,

    /// Format used to record the item.
    pub format: String,

    /// Burn record version used to record the item.
    pub version: String,

    /// Settings used to record the item.
    pub settings: String,
}

/// Record that can be saved by a [Recorder](Recorder).
#[derive(Serialize, Deserialize, Debug)]
pub struct BurnRecord<I, B: Backend> {
    /// Metadata of the record.
    pub metadata: BurnMetadata,

    /// Item to record.
    pub item: I,

    _b: PhantomData<B>,
}

impl<I, B: Backend> BurnRecord<I, B> {
    /// Creates a new record.
    ///
    /// # Arguments
    ///
    /// * `item` - Item to record.
    ///
    /// # Returns
    ///
    /// The new record.
    pub fn new<R: Recorder<B>>(item: I) -> Self {
        let metadata = recorder_metadata::<R, B>();

        Self {
            metadata,
            item,
            _b: PhantomData,
        }
    }
}

/// Record that can be saved by a [Recorder](Recorder) without the item.
#[derive(new, Debug, Serialize, Deserialize)]
pub struct BurnRecordNoItem {
    /// Metadata of the record.
    pub metadata: BurnMetadata,
}

/// Default recorder.
///
/// It uses the [named msgpack](rmp_serde) format for serialization with full precision.
#[cfg(feature = "std")]
pub type DefaultRecorder = DefaultFileRecorder<FullPrecisionSettings>;

/// Recorder optimized for compactness.
///
/// It uses the [named msgpack](rmp_serde) format for serialization with half precision.
/// If you are looking for the recorder that offers the smallest file size, have a look at
/// [sensitive compact recorder](SensitiveCompactRecorder).
#[cfg(feature = "std")]
pub type CompactRecorder = DefaultFileRecorder<HalfPrecisionSettings>;

/// Recorder optimized for compactness making it a good choice for model deployment.
///
/// It uses the [bincode](bincode) format for serialization and half precision.
/// This format is not resilient to type changes since no metadata is encoded.
/// Favor [default recorder](DefaultRecorder) or [compact recorder](CompactRecorder)
/// for long term data storage.
#[cfg(feature = "std")]
pub type SensitiveCompactRecorder = BinGzFileRecorder<HalfPrecisionSettings>;

/// Training recorder compatible with no-std inference.
#[cfg(feature = "std")]
pub type NoStdTrainingRecorder = BinFileRecorder<FullPrecisionSettings>;

/// Inference recorder compatible with no-std.
pub type NoStdInferenceRecorder = BinBytesRecorder<FullPrecisionSettings, &'static [u8]>;

/// Debug recorder.
///
/// It uses the [pretty json](serde_json) format for serialization with full precision making it
/// human readable.
#[cfg(feature = "std")]
pub type DebugRecordSettings = PrettyJsonFileRecorder<FullPrecisionSettings>;

#[cfg(all(test, feature = "std"))]
mod tests {
    static FILE_PATH: &str = "/tmp/burn_test_record";

    use crate::TestBackend;

    use super::*;
    use burn_tensor::{Device, ElementConversion};

    #[test]
    #[should_panic]
    fn err_when_invalid_item() {
        #[derive(new, Serialize, Deserialize, Clone)]
        struct Item<S: PrecisionSettings> {
            value: S::FloatElem,
        }

        impl<D, B> Record<B> for Item<D>
        where
            D: PrecisionSettings,
            B: Backend,
        {
            type Item<S: PrecisionSettings> = Item<S>;

            fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
                Item {
                    value: self.value.elem(),
                }
            }

            fn from_item<S: PrecisionSettings>(item: Self::Item<S>, _device: &B::Device) -> Self {
                Item {
                    value: item.value.elem(),
                }
            }
        }

        let item = Item::<FullPrecisionSettings>::new(16.elem());
        let device: Device<TestBackend> = Default::default();

        // Serialize in f32.
        let recorder = DefaultFileRecorder::<FullPrecisionSettings>::new();
        Recorder::<TestBackend>::record(&recorder, item, FILE_PATH.into()).unwrap();

        // Can't deserialize f32 into f16.
        let recorder = DefaultFileRecorder::<HalfPrecisionSettings>::new();
        Recorder::<TestBackend>::load::<Item<FullPrecisionSettings>>(
            &recorder,
            FILE_PATH.into(),
            &device,
        )
        .unwrap();
    }
}
