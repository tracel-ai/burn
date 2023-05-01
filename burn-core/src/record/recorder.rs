use alloc::format;
use alloc::string::String;
use serde::{Deserialize, Serialize};

use super::{Record, RecordSettings};

/// Record any item implementing [Serialize](Serialize) and [DeserializeOwned](DeserializeOwned).
pub trait Recorder: Send + Sync + core::default::Default + core::fmt::Debug + Clone {
    type Settings: RecordSettings;
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
            core::any::type_name::<<Self::Settings as RecordSettings>::FloatElem>().to_string(),
            core::any::type_name::<<Self::Settings as RecordSettings>::IntElem>().to_string(),
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

// #[cfg(all(test, feature = "std"))]
// mod tests {
//     static FILE_PATH: &str = "/tmp/burn_test_record";
//
//     use core::marker::PhantomData;
//
//     use super::*;
//     use crate::record::FileJsonGzRecorder;
//     use burn_tensor::{Element, ElementConversion};
//
//     #[test]
//     #[should_panic]
//     fn err_when_invalid_object() {
//         #[derive(Debug, Default)]
//         pub struct TestSettings<F> {
//             float: PhantomData<F>,
//         }
//
//         impl<F: Element + Serialize + DeserializeOwned> RecordSettings for TestSettings<F> {
//             type FloatElem = F;
//             type IntElem = i32;
//             type Recorder = FileJsonGzRecorder;
//         }
//
//         #[derive(new, Serialize, Deserialize)]
//         struct Item<S: RecordSettings> {
//             value: S::FloatElem,
//         }
//
//         impl<D: RecordSettings> Record for Item<D> {
//             type Item<S: RecordSettings> = Item<S>;
//
//             fn into_item<S: RecordSettings>(self) -> Self::Item<S> {
//                 Item {
//                     value: self.value.elem(),
//                 }
//             }
//
//             fn from_item<S: RecordSettings>(item: Self::Item<S>) -> Self {
//                 Item {
//                     value: item.value.elem(),
//                 }
//             }
//         }
//
//         let item = Item::<TestSettings<f32>>::new(16.elem());
//
//         // Serialize in f32.
//         item.record::<TestSettings<f32>>(FILE_PATH.into()).unwrap();
//         // Can't deserialize u8 into f32.
//         Item::<TestSettings<f32>>::load::<TestSettings<u8>>(FILE_PATH.into()).unwrap();
//     }
// }
