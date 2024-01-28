use alloc::{
    string::{String, ToString},
    vec,
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

impl<B> Record<B> for ()
where
    B: Backend,
{
    type Item<S: PrecisionSettings> = ();

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {}

    fn from_item<S: PrecisionSettings>(_item: Self::Item<S>, _device: &B::Device) -> Self {}
}

impl<T, B> Record<B> for Vec<T>
where
    T: Record<B>,
    B: Backend,
{
    type Item<S: PrecisionSettings> = Vec<T::Item<S>>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        self.into_iter().map(Record::into_item).collect()
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        item.into_iter()
            .map(|i| Record::from_item(i, device))
            .collect()
    }
}

impl<T, B> Record<B> for Option<T>
where
    T: Record<B>,
    B: Backend,
{
    type Item<S: PrecisionSettings> = Option<T::Item<S>>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        self.map(Record::into_item)
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        item.map(|i| Record::from_item(i, device))
    }
}

impl<const N: usize, T, B> Record<B> for [T; N]
where
    T: Record<B>,
    B: Backend,
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

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        item.0.map(|i| Record::from_item(i, device))
    }
}

impl<T, B> Record<B> for HashMap<ParamId, T>
where
    T: Record<B>,
    B: Backend,
{
    type Item<S: PrecisionSettings> = HashMap<String, T::Item<S>>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        let mut items = HashMap::with_capacity(self.len());
        self.into_iter().for_each(|(id, record)| {
            items.insert(id.to_string(), record.into_item());
        });
        items
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        let mut record = HashMap::with_capacity(item.len());
        item.into_iter().for_each(|(id, item)| {
            record.insert(ParamId::from(id), T::from_item(item, device));
        });
        record
    }
}

impl<E, B> Record<B> for DataSerialize<E>
where
    E: Element,
    B: Backend,
{
    type Item<S: PrecisionSettings> = DataSerialize<S::FloatElem>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        self.convert()
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, _device: &B::Device) -> Self {
        item.convert()
    }
}

/// (De)serialize parameters into a clean format.
#[derive(new, Debug, Clone, Serialize, Deserialize)]
pub struct ParamSerde<T> {
    id: String,
    param: T,
}

impl<B, const D: usize> Record<B> for Param<Tensor<B, D>>
where
    B: Backend,
{
    type Item<S: PrecisionSettings> = ParamSerde<FloatTensorSerde<S>>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        ParamSerde::new(self.id.into_string(), self.value.into_item())
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        Param::new(
            ParamId::from(item.id),
            Tensor::from_item(item.param, device).require_grad(), // Same behavior as when we create a new
                                                                  // Param from a tensor.
        )
    }
}

impl<B, const D: usize> Record<B> for Param<Tensor<B, D, Int>>
where
    B: Backend,
{
    type Item<S: PrecisionSettings> = ParamSerde<IntTensorSerde<S>>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        ParamSerde::new(self.id.into_string(), self.value.into_item())
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        Param::new(
            ParamId::from(item.id),
            Tensor::from_item(item.param, device),
        )
    }
}

impl<B, const D: usize> Record<B> for Param<Tensor<B, D, Bool>>
where
    B: Backend,
{
    type Item<S: PrecisionSettings> = ParamSerde<BoolTensorSerde>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        ParamSerde::new(self.id.into_string(), self.value.into_item::<S>())
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        Param::new(
            ParamId::from(item.id),
            Tensor::from_item::<S>(item.param, device),
        )
    }
}

// Type that can be serialized as is without any conversion.
macro_rules! primitive {
    ($type:ty) => {
        impl<B: Backend> Record<B> for $type {
            type Item<S: PrecisionSettings> = $type;

            fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
                self
            }

            fn from_item<S: PrecisionSettings>(item: Self::Item<S>, _device: &B::Device) -> Self {
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
                    .map_err(|_| "An array of size {N}")
                    .unwrap();

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
