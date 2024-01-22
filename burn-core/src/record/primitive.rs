use alloc::{
    string::{String, ToString},
    vec::Vec,
};
use core::{fmt, marker::PhantomData};

use super::tensor::{BoolTensorSerde, FloatTensorSerde, IntTensorSerde};
use super::{PrecisionSettings, Record};
use crate::module::{Param, ParamId};

use burn_tensor::{backend::Backend, Bool, DataSerialize, Element, Int, Tensor};

use hashbrown::HashMap;
use serde::{
    de::{Error, SeqAccess, Visitor},
    ser::SerializeTuple,
    Deserialize, Serialize,
};

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

impl<const N: usize, T> Record for [T; N]
where
    T: Record,
{
    /// The record item is an array of the record item of the elements.
    /// The reason why we wrap the array in a struct is because serde does not support
    /// deserializing arrays of variable size,
    /// see [serde/issues/1937](https://github.com/serde-rs/serde/issues/1937).
    /// for backward compatibility reasons. Serde APIs were created before const generics.
    type Item<S: PrecisionSettings> = Array<N, T::Item<S>>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        Array(self.map(Record::into_item))
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>) -> Self {
        item.0.map(Record::from_item)
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
    type Item<S: PrecisionSettings> = ParamSerde<FloatTensorSerde<S>>;

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

impl<B: Backend, const D: usize> Record for Param<Tensor<B, D, Int>> {
    type Item<S: PrecisionSettings> = ParamSerde<IntTensorSerde<S>>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        ParamSerde::new(self.id.into_string(), self.value.into_item())
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>) -> Self {
        Param::new(ParamId::from(item.id), Tensor::from_item(item.param))
    }
}

impl<B: Backend, const D: usize> Record for Param<Tensor<B, D, Bool>> {
    type Item<S: PrecisionSettings> = ParamSerde<BoolTensorSerde>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        ParamSerde::new(self.id.into_string(), self.value.into_item::<S>())
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>) -> Self {
        Param::new(ParamId::from(item.id), Tensor::from_item::<S>(item.param))
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

/// A wrapper around an array of size N, so that it can be serialized and deserialized
/// using serde.
///
/// The reason why we wrap the array in a struct is because serde does not support
/// deserializing arrays of variable size,
/// see [serde/issues/1937](https://github.com/serde-rs/serde/issues/1937)
/// for backward compatibility reasons. Serde APIs were created before const generics.
pub struct Array<const N: usize, T>([T; N]);

impl<T: Serialize, const N: usize> Serialize for Array<N, T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut seq = serializer.serialize_tuple(self.0.len())?;
        for element in &self.0 {
            seq.serialize_element(element)?;
        }
        seq.end()
    }
}

impl<'de, T, const N: usize> Deserialize<'de> for Array<N, T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct ArrayVisitor<T, const N: usize> {
            marker: PhantomData<T>,
        }

        impl<'de, T, const N: usize> Visitor<'de> for ArrayVisitor<T, N>
        where
            T: Deserialize<'de>,
        {
            type Value = Array<N, T>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a fixed size array")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut items = vec![];

                for i in 0..N {
                    let item = seq
                        .next_element()?
                        .ok_or_else(|| Error::invalid_length(i, &self))?;
                    items.push(item);
                }

                let array: [T; N] = items
                    .into_iter()
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap_or_else(|_| panic!("An arrar of size {N}"));

                Ok(Array(array))
            }
        }

        deserializer.deserialize_tuple(
            N,
            ArrayVisitor {
                marker: PhantomData,
            },
        )
    }
}
