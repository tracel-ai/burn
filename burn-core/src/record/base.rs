use super::{RecordSettings, Recorder, RecorderError};
use crate::{alloc::string::ToString, module::Param};
use alloc::format;
use alloc::string::String;
use burn_tensor::{backend::Backend, Bool, Data, DataSerialize, Element, Int, Tensor};
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
            format!("{:?}", S::default()),
        );
        let item = self.into_item::<S>();
        let record = BurnRecord::new(item, metadata);

        RecorderType::<S>::record(record, args)
    }

    /// Load the record using the given [settings](RecordSettings).
    fn load<S>(args: LoadArgs<S>) -> Result<Self, RecorderError>
    where
        Self: Sized,
        S: RecordSettings,
        Self::Item<S>: Serialize + DeserializeOwned,
    {
        let record: BurnRecord<Self::Item<S>> =
            RecorderType::<S>::load(args.clone()).map_err(|err| {
                let message = match err {
                    RecorderError::FileNotFound(_) => return err,
                    RecorderError::Unknown(message) => message,
                };
                let record = RecorderType::<S>::load::<BurnRecordNoItem>(args);

                let message = match record {
                    Ok(record) => format!(
                        "Unable to load record with settings {:?}, found metadata {:?}, err: {:?}",
                        S::default(),
                        record.metadata,
                        message
                    ),
                    Err(_err) => message,
                };
                RecorderError::Unknown(message)
            })?;

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
/// Recorder for the given settings.
pub type RecorderType<S> = <S as RecordSettings>::Recorder;

#[derive(new, Debug, Serialize, Deserialize)]
struct BurnMetadata {
    float: String,
    int: String,
    format: String,
    version: String,
    settings: String,
}

#[derive(new, Serialize, Deserialize)]
struct BurnRecord<I> {
    item: I,
    metadata: BurnMetadata,
}

#[derive(new, Serialize, Deserialize)]
struct BurnRecordNoItem {
    metadata: BurnMetadata,
}

impl Record for () {
    type Item<S: RecordSettings> = ();

    fn into_item<S: RecordSettings>(self) -> Self::Item<S> {}

    fn from_item<S: RecordSettings>(_item: Self::Item<S>) -> Self {}
}

impl<T: Record> Record for Vec<T> {
    type Item<S: RecordSettings> = Vec<T::Item<S>>;

    fn into_item<S: RecordSettings>(self) -> Self::Item<S> {
        self.into_iter().map(Record::into_item).collect()
    }

    fn from_item<S: RecordSettings>(item: Self::Item<S>) -> Self {
        item.into_iter().map(Record::from_item).collect()
    }
}

impl<T: Record> Record for Option<T> {
    type Item<S: RecordSettings> = Option<T::Item<S>>;

    fn into_item<S: RecordSettings>(self) -> Self::Item<S> {
        self.map(Record::into_item)
    }

    fn from_item<S: RecordSettings>(item: Self::Item<S>) -> Self {
        item.map(Record::from_item)
    }
}

impl<const N: usize, T: Record> Record for [T; N] {
    type Item<S: RecordSettings> = [T::Item<S>; N];

    fn into_item<S: RecordSettings>(self) -> Self::Item<S> {
        self.map(Record::into_item)
    }

    fn from_item<S: RecordSettings>(item: Self::Item<S>) -> Self {
        item.map(Record::from_item)
    }
}

impl<E: Element> Record for DataSerialize<E> {
    type Item<S: RecordSettings> = DataSerialize<S::FloatElem>;

    fn into_item<S: RecordSettings>(self) -> Self::Item<S> {
        self.convert()
    }

    fn from_item<S: RecordSettings>(item: Self::Item<S>) -> Self {
        item.convert()
    }
}

impl<B: Backend, const D: usize> Record for Tensor<B, D> {
    type Item<S: RecordSettings> = DataSerialize<S::FloatElem>;

    fn into_item<S: RecordSettings>(self) -> Self::Item<S> {
        self.into_data().serialize().convert()
    }

    fn from_item<S: RecordSettings>(item: Self::Item<S>) -> Self {
        Tensor::from_data(Data::<S::FloatElem, D>::from(item).convert::<B::FloatElem>())
    }
}

impl<B: Backend, const D: usize> Record for Tensor<B, D, Int> {
    type Item<S: RecordSettings> = DataSerialize<S::IntElem>;

    fn into_item<S: RecordSettings>(self) -> Self::Item<S> {
        self.into_data().serialize().convert()
    }

    fn from_item<S: RecordSettings>(item: Self::Item<S>) -> Self {
        Tensor::from_data(Data::<S::IntElem, D>::from(item).convert::<B::IntElem>())
    }
}

impl<B: Backend, const D: usize> Record for Tensor<B, D, Bool> {
    type Item<S: RecordSettings> = DataSerialize<bool>;

    fn into_item<S: RecordSettings>(self) -> Self::Item<S> {
        self.into_data().serialize()
    }

    fn from_item<S: RecordSettings>(item: Self::Item<S>) -> Self {
        Tensor::from_data(Data::from(item))
    }
}

impl<T: Record> Record for Param<T> {
    type Item<S: RecordSettings> = Param<T::Item<S>>;

    fn into_item<S: RecordSettings>(self) -> Self::Item<S> {
        Param {
            id: self.id,
            value: self.value.into_item(),
        }
    }

    fn from_item<S: RecordSettings>(item: Self::Item<S>) -> Self {
        Param {
            id: item.id,
            value: T::from_item(item.value),
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    static FILE_PATH: &str = "/tmp/burn_test_record";

    use core::marker::PhantomData;

    use super::*;
    use crate::record::FileJsonGzRecorder;
    use burn_tensor::{Element, ElementConversion};

    #[test]
    #[should_panic]
    fn err_when_invalid_object() {
        #[derive(Debug, Default)]
        pub struct TestSettings<F> {
            float: PhantomData<F>,
        }

        impl<F: Element + Serialize + DeserializeOwned> RecordSettings for TestSettings<F> {
            type FloatElem = F;
            type IntElem = i32;
            type Recorder = FileJsonGzRecorder;
        }

        #[derive(new, Serialize, Deserialize)]
        struct Item<S: RecordSettings> {
            value: S::FloatElem,
        }

        impl<D: RecordSettings> Record for Item<D> {
            type Item<S: RecordSettings> = Item<S>;

            fn into_item<S: RecordSettings>(self) -> Self::Item<S> {
                Item {
                    value: self.value.elem(),
                }
            }

            fn from_item<S: RecordSettings>(item: Self::Item<S>) -> Self {
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
