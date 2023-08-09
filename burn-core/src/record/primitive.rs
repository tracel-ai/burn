use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;
use burn_tensor::backend::Backend;
use burn_tensor::Tensor;
use serde::Deserialize;
use serde::Serialize;

use super::tensor::FloatTensorSerde;
use super::{PrecisionSettings, Record};
use crate::module::{Param, ParamId};
use burn_tensor::{DataSerialize, Element};
use hashbrown::HashMap;

impl Record for () {
    type Item<S: PrecisionSettings> = ();

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {}

    fn from_item<S: PrecisionSettings>(_item: Self::Item<S>) -> Self {}
}

impl<T: Record> Record for Vec<T> {
    type Item<S: PrecisionSettings> = Vec<T::Item<S>>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        self.into_iter().map(Record::into_item).collect()
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>) -> Self {
        item.into_iter().map(Record::from_item).collect()
    }
}

impl<T: Record> Record for Option<T> {
    type Item<S: PrecisionSettings> = Option<T::Item<S>>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        self.map(Record::into_item)
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>) -> Self {
        item.map(Record::from_item)
    }
}

impl<const N: usize, T: Record + core::fmt::Debug> Record for [T; N] {
    type Item<S: PrecisionSettings> = Vec<T::Item<S>>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        self.map(Record::into_item).into_iter().collect()
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>) -> Self {
        item.into_iter()
            .map(Record::from_item)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap_or_else(|_| panic!("An arrar of size {N}"))
    }
}

impl<T: Record> Record for HashMap<ParamId, T> {
    type Item<S: PrecisionSettings> = HashMap<String, T::Item<S>>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        let mut items = HashMap::with_capacity(self.len());
        self.into_iter().for_each(|(id, record)| {
            items.insert(id.to_string(), record.into_item());
        });
        items
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>) -> Self {
        let mut record = HashMap::with_capacity(item.len());
        item.into_iter().for_each(|(id, item)| {
            record.insert(ParamId::from(id), T::from_item(item));
        });
        record
    }
}

impl<E: Element> Record for DataSerialize<E> {
    type Item<S: PrecisionSettings> = DataSerialize<S::FloatElem>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        self.convert()
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>) -> Self {
        item.convert()
    }
}

/// (De)serialize parameters into a clean format.
#[derive(new, Debug, Clone, Serialize, Deserialize)]
pub struct ParamSerde<T> {
    id: String,
    param: T,
}

impl<B: Backend, const D: usize> Record for Param<Tensor<B, D>> {
    type Item<S: PrecisionSettings> = ParamSerde<FloatTensorSerde<B, D, S>>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        ParamSerde::new(self.id.into_string(), self.value.into_item())
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>) -> Self {
        Param::new(
            ParamId::from(item.id),
            Tensor::from_item(item.param).require_grad(), // Same behavior as when we create a new
                                                          // Param from a tensor.
        )
    }
}

// Type that can be serialized as is without any conversion.
macro_rules! primitive {
    ($type:ty) => {
        impl Record for $type {
            type Item<S: PrecisionSettings> = $type;

            fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
                self
            }

            fn from_item<S: PrecisionSettings>(item: Self::Item<S>) -> Self {
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

primitive!(half::bf16);
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
