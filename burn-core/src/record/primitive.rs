use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;
use burn_tensor::backend::Backend;
use burn_tensor::Bool;
use burn_tensor::Int;
use burn_tensor::Tensor;
use serde::Deserialize;
use serde::Serialize;

use super::tensor::BoolTensorSerde;
use super::tensor::FloatTensorSerde;
use super::tensor::IntTensorSerde;
use super::{PrecisionSettings, Record};
use crate::module::{Param, ParamId};
use burn_tensor::{DataSerialize, Element};
use hashbrown::HashMap;

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
    T: Record<B> + core::fmt::Debug,
    B: Backend,
{
    type Item<S: PrecisionSettings> = Vec<T::Item<S>>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        self.map(Record::into_item).into_iter().collect()
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        item.into_iter()
            .map(|i| Record::from_item(i, device))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap_or_else(|_| panic!("An arrar of size {N}"))
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

            fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
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
