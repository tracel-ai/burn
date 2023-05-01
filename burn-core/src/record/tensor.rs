use super::{PrecisionSettings, Record};
use burn_tensor::{backend::Backend, Bool, DataSerialize, Int, Tensor};
use core::marker::PhantomData;
use serde::{Deserialize, Serialize};

/// This struct implements serde to lazily serialize and deserialize a float tensor
/// using the given [record settings](RecordSettings).
#[derive(new, Clone, Debug)]
pub struct FloatTensorSerde<B: Backend, const D: usize, S: PrecisionSettings> {
    tensor: Tensor<B, D>,
    elem: PhantomData<S>,
}

/// This struct implements serde to lazily serialize and deserialize an int tensor
/// using the given [record settings](RecordSettings).
#[derive(new, Clone, Debug)]
pub struct IntTensorSerde<B: Backend, const D: usize, S: PrecisionSettings> {
    tensor: Tensor<B, D, Int>,
    elem: PhantomData<S>,
}

/// This struct implements serde to lazily serialize and deserialize an bool tensor.
#[derive(new, Clone, Debug)]
pub struct BoolTensorSerde<B: Backend, const D: usize> {
    tensor: Tensor<B, D, Bool>,
}

// --- SERDE IMPLEMENTATIONS --- //

impl<B: Backend, const D: usize, S: PrecisionSettings> Serialize for FloatTensorSerde<B, D, S> {
    fn serialize<Se>(&self, serializer: Se) -> Result<Se::Ok, Se::Error>
    where
        Se: serde::Serializer,
    {
        self.tensor
            .to_data()
            .convert::<S::FloatElem>()
            .serialize()
            .serialize(serializer)
    }
}

impl<'de, B: Backend, const D: usize, S: PrecisionSettings> Deserialize<'de>
    for FloatTensorSerde<B, D, S>
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: serde::Deserializer<'de>,
    {
        let data = DataSerialize::<S::FloatElem>::deserialize(deserializer)?;
        let tensor = Tensor::from_data(data.convert::<B::FloatElem>());

        Ok(Self::new(tensor))
    }
}

impl<B: Backend, const D: usize, S: PrecisionSettings> Serialize for IntTensorSerde<B, D, S> {
    fn serialize<Se>(&self, serializer: Se) -> Result<Se::Ok, Se::Error>
    where
        Se: serde::Serializer,
    {
        self.tensor
            .to_data()
            .convert::<S::IntElem>()
            .serialize()
            .serialize(serializer)
    }
}

impl<'de, B: Backend, const D: usize, S: PrecisionSettings> Deserialize<'de>
    for IntTensorSerde<B, D, S>
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: serde::Deserializer<'de>,
    {
        let data = DataSerialize::<S::IntElem>::deserialize(deserializer)?;
        let tensor = Tensor::from_data(data.convert::<B::IntElem>());

        Ok(Self::new(tensor))
    }
}

impl<B: Backend, const D: usize> Serialize for BoolTensorSerde<B, D> {
    fn serialize<Se>(&self, serializer: Se) -> Result<Se::Ok, Se::Error>
    where
        Se: serde::Serializer,
    {
        self.tensor.to_data().serialize().serialize(serializer)
    }
}

impl<'de, B: Backend, const D: usize> Deserialize<'de> for BoolTensorSerde<B, D> {
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: serde::Deserializer<'de>,
    {
        let data = DataSerialize::<bool>::deserialize(deserializer)?;
        let tensor = Tensor::from_data(data);

        Ok(Self::new(tensor))
    }
}

// --- RECORD IMPLEMENTATIONS --- //

impl<B: Backend, const D: usize> Record for Tensor<B, D> {
    type Item<S: PrecisionSettings> = FloatTensorSerde<B, D, S>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        FloatTensorSerde::new(self)
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>) -> Self {
        item.tensor
    }
}

impl<B: Backend, const D: usize> Record for Tensor<B, D, Int> {
    type Item<S: PrecisionSettings> = IntTensorSerde<B, D, S>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        IntTensorSerde::new(self)
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>) -> Self {
        item.tensor
    }
}

impl<B: Backend, const D: usize> Record for Tensor<B, D, Bool> {
    type Item<S: PrecisionSettings> = BoolTensorSerde<B, D>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        BoolTensorSerde::new(self)
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>) -> Self {
        item.tensor
    }
}
