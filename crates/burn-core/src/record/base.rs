pub use burn_derive::Record;
use burn_tensor::backend::Backend;

use super::PrecisionSettings;
use serde::{Serialize, de::DeserializeOwned};

/// Trait to define a family of types which can be recorded using any [settings](PrecisionSettings).
pub trait Record<B: Backend>: Send {
    /// Type of the item that can be serialized and deserialized.
    type Item<S: PrecisionSettings>: Serialize + DeserializeOwned + Clone;

    /// Convert the current record into the corresponding item that follows the given [settings](PrecisionSettings).
    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S>;

    /// Convert the given item into a record.
    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self;
}
