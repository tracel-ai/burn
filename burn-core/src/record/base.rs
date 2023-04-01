use super::RecordSettings;
use alloc::string::String;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

/// Record any objects implementing [Serialize](Serialize) and [DeserializeOwned](DeserializeOwned).
pub trait Recorder {
    /// Arguments used to record objects.
    type RecordArgs;
    /// Record output type.
    type RecordOutput;
    /// Arguments used to load recorded objects.
    type LoadArgs;

    fn record<Item: Serialize + DeserializeOwned>(
        item: Item,
        args: Self::RecordArgs,
    ) -> Result<Self::RecordOutput, RecordError>;
    /// Load an object from the given arguments.
    fn load<Item: Serialize + DeserializeOwned>(args: Self::LoadArgs) -> Result<Item, RecordError>;
}

pub type RecorderSaveArgs<S> = <<S as RecordSettings>::Recorder as Recorder>::RecordArgs;
pub type RecorderLoadArgs<S> = <<S as RecordSettings>::Recorder as Recorder>::LoadArgs;
pub type RecorderSaveResult<S> =
    Result<<<S as RecordSettings>::Recorder as Recorder>::RecordOutput, RecordError>;

pub trait Record {
    type Item<S: RecordSettings>;

    fn into_item<S: RecordSettings>(self) -> Self::Item<S>;
    fn from_item<S: RecordSettings>(item: Self::Item<S>) -> Self;

    /// Save the record to the provided file path using the given [StateFormat](StateFormat).
    ///
    /// # Notes
    ///
    /// The file extension will be added automatically depending on the state format.
    fn save<S>(self, args: RecorderSaveArgs<S>) -> RecorderSaveResult<S>
    where
        Self: Sized,
        S: RecordSettings,
        Self::Item<S>: Serialize + DeserializeOwned,
    {
        RecordWrapper::<Self>::save(self, args)
    }

    /// Load the record from the provided file path using the given [StateFormat](StateFormat).
    ///
    /// # Notes
    ///
    /// The file extension will be added automatically depending on the state format.
    fn load<S>(args: RecorderLoadArgs<S>) -> Result<Self, RecordError>
    where
        Self: Sized,
        S: RecordSettings,
        Self::Item<S>: Serialize + DeserializeOwned,
    {
        RecordWrapper::load(args)
    }
}

#[derive(Debug)]
pub enum RecordError {
    FileNotFound(String),
    Unknown(String),
}

impl core::fmt::Display for RecordError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(format!("{self:?}").as_str())
    }
}

// TODO: Move from std to core after Error is core (see https://github.com/rust-lang/rust/issues/103765)
#[cfg(feature = "std")]
impl std::error::Error for RecordError {}

#[derive(new, Serialize, Deserialize)]
struct Metadata {
    elem_float: String,
    elem_int: String,
    format: String,
    version: String,
}

#[derive(Serialize, Deserialize)]
struct RecordWrapper<R> {
    record: R,
    metadata: Metadata,
}

impl<R> RecordWrapper<R>
where
    R: Record,
{
    fn save<S>(record: R, args: RecorderSaveArgs<S>) -> RecorderSaveResult<S>
    where
        S: RecordSettings,
        R::Item<S>: Serialize + DeserializeOwned,
    {
        let metadata = Metadata::new(
            core::any::type_name::<S::FloatElem>().to_string(),
            core::any::type_name::<S::IntElem>().to_string(),
            core::any::type_name::<S::Recorder>().to_string(),
            env!("CARGO_PKG_VERSION").to_string(),
        );
        let item = record.into_item::<S>();
        let record = RecordWrapper {
            record: item,
            metadata,
        };

        <S::Recorder as Recorder>::record(record, args)
    }

    fn load<S>(args: RecorderLoadArgs<S>) -> Result<R, RecordError>
    where
        S: RecordSettings,
        R::Item<S>: Serialize + DeserializeOwned,
    {
        let record: RecordWrapper<R::Item<S>> = <S::Recorder as Recorder>::load(args)?;

        Ok(R::from_item(record.record))
    }
}

pub(crate) fn bin_config() -> bincode::config::Configuration {
    bincode::config::standard()
}
