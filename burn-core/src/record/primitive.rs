use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;

use super::{Record, RecordSettings};
use crate::module::{Param, ParamId};
use burn_tensor::{DataSerialize, Element};
use hashbrown::HashMap;

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

impl<const N: usize, T: Record + core::fmt::Debug> Record for [T; N] {
    type Item<S: RecordSettings> = Vec<T::Item<S>>;

    fn into_item<S: RecordSettings>(self) -> Self::Item<S> {
        self.map(Record::into_item).into_iter().collect()
    }

    fn from_item<S: RecordSettings>(item: Self::Item<S>) -> Self {
        item.into_iter()
            .map(Record::from_item)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap_or_else(|_| panic!("An arrar of size {N}"))
    }
}

impl<T: Record> Record for HashMap<ParamId, T> {
    type Item<S: RecordSettings> = HashMap<String, T::Item<S>>;

    fn into_item<S: RecordSettings>(self) -> Self::Item<S> {
        let mut items = HashMap::with_capacity(self.len());
        self.into_iter().for_each(|(id, record)| {
            items.insert(id.to_string(), record.into_item());
        });
        items
    }

    fn from_item<S: RecordSettings>(item: Self::Item<S>) -> Self {
        let mut record = HashMap::with_capacity(item.len());
        item.into_iter().for_each(|(id, item)| {
            record.insert(ParamId::from(id), T::from_item(item));
        });
        record
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

// Type that can be serialized as is without any convertion.
macro_rules! primitive {
    ($type:ty) => {
        impl Record for $type {
            type Item<S: RecordSettings> = $type;

            fn into_item<S: RecordSettings>(self) -> Self::Item<S> {
                self
            }

            fn from_item<S: RecordSettings>(item: Self::Item<S>) -> Self {
                item
            }
        }
    };
}

// General Types
primitive!(alloc::string::String);
primitive!(bool);

// Float Types
primitive!(f64);
primitive!(f32);

// TODO: Remove the feature flag when half supports serde with no_std
#[cfg(feature = "std")]
primitive!(half::bf16);
#[cfg(feature = "std")]
primitive!(half::f16);

// Unsigned Integer Types
primitive!(usize);
primitive!(u64);
primitive!(u32);
primitive!(u16);
primitive!(u8);

// Signed Integer Types
primitive!(i64);
primitive!(i32);
primitive!(i16);
primitive!(i8);
