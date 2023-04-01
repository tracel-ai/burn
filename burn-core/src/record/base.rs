use super::{RecordSettings, Recorder, RecorderError};
use alloc::string::String;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

/// Trait to define a family of types which can be recorded using any [settings](RecordSettings).
pub trait Record: Send + Sync {
    type Item<S: RecordSettings>;

    /// Convert the current record into the corresponding item that follows the given [settings](RecordSettings).
    fn into_item<S: RecordSettings>(self) -> Self::Item<S>;
    /// Convert the given item into a record.
    fn from_item<S: RecordSettings>(item: Self::Item<S>) -> Self;

    /// Record using the given [settings](RecordSettings).
    fn record<S>(self, args: RecordArgs<S>) -> RecordOutputResult<S>
    where
        Self: Sized,
        S: RecordSettings,
        Self::Item<S>: Serialize + DeserializeOwned,
    {
        let metadata = BurnMetadata::new(
            core::any::type_name::<S::FloatElem>().to_string(),
            core::any::type_name::<S::IntElem>().to_string(),
            core::any::type_name::<S::Recorder>().to_string(),
            env!("CARGO_PKG_VERSION").to_string(),
        );
        let item = self.into_item::<S>();
        let record = BurnRecord::new(item, metadata);

        <S::Recorder as Recorder>::record(record, args)
    }

    /// Load the record using the given [settings](RecordSettings).
    fn load<S>(args: LoadArgs<S>) -> Result<Self, RecorderError>
    where
        Self: Sized,
        S: RecordSettings,
        Self::Item<S>: Serialize + DeserializeOwned,
    {
        let record: BurnRecord<Self::Item<S>> = <S::Recorder as Recorder>::load(args)?;

        Ok(Self::from_item(record.item))
    }
}

/// Record arguments for the given settings.
pub type RecordArgs<S> = <<S as RecordSettings>::Recorder as Recorder>::RecordArgs;
/// Record loading arguments for the given settings.
pub type LoadArgs<S> = <<S as RecordSettings>::Recorder as Recorder>::LoadArgs;
/// Record output result for the given settings.
pub type RecordOutputResult<S> =
    Result<<<S as RecordSettings>::Recorder as Recorder>::RecordOutput, RecorderError>;

#[derive(new, Serialize, Deserialize)]
struct BurnMetadata {
    float: String,
    int: String,
    format: String,
    version: String,
}

#[derive(new, Serialize, Deserialize)]
struct BurnRecord<I> {
    item: I,
    metadata: BurnMetadata,
}
