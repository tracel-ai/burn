pub use burn_derive::Record;

use super::RecordSettings;
use serde::{de::DeserializeOwned, Serialize};

/// Trait to define a family of types which can be recorded using any [settings](RecordSettings).
pub trait Record: Send + Sync {
    type Item<S: RecordSettings>: Serialize + DeserializeOwned;

    /// Convert the current record into the corresponding item that follows the given [settings](RecordSettings).
    fn into_item<S: RecordSettings>(self) -> Self::Item<S>;
    /// Convert the given item into a record.
    fn from_item<S: RecordSettings>(item: Self::Item<S>) -> Self;
}
