use super::{Record, RecordSettings};
use burn_tensor::{backend::Backend, Bool, Data, DataSerialize, Int, Tensor};
use core::marker::PhantomData;
use serde::{Deserialize, Serialize};

#[derive(new, Clone, Debug)]
pub struct FloatTensorSerde<B: Backend, const D: usize, S: RecordSettings> {
    tensor: Tensor<B, D>,
    elem: PhantomData<S>,
}
#[derive(new, Clone, Debug)]
pub struct IntTensorSerde<B: Backend, const D: usize, S: RecordSettings> {
    tensor: Tensor<B, D, Int>,
    elem: PhantomData<S>,
}

impl<B: Backend, const D: usize, S: RecordSettings> Serialize for FloatTensorSerde<B, D, S> {
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

impl<'de, B: Backend, const D: usize, S: RecordSettings> Deserialize<'de>
    for FloatTensorSerde<B, D, S>
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: serde::Deserializer<'de>,
    {
        let data = DataSerialize::<S::FloatElem>::deserialize(deserializer)?;
        let tensor = Tensor::from_data(data.convert::<B::FloatElem>().into());

        Ok(Self::new(tensor))
    }
}

impl<B: Backend, const D: usize, S: RecordSettings> Serialize for IntTensorSerde<B, D, S> {
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

impl<'de, B: Backend, const D: usize, S: RecordSettings> Deserialize<'de>
    for IntTensorSerde<B, D, S>
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: serde::Deserializer<'de>,
    {
        let data = DataSerialize::<S::FloatElem>::deserialize(deserializer)?;
        let tensor = Tensor::from_data(data.convert::<B::IntElem>().into());

        Ok(Self::new(tensor))
    }
}

impl<B: Backend, const D: usize> Record for Tensor<B, D> {
    type Item<S: RecordSettings> = FloatTensorSerde<B, D, S>;

    fn into_item<S: RecordSettings>(self) -> Self::Item<S> {
        FloatTensorSerde::new(self)
    }

    fn from_item<S: RecordSettings>(item: Self::Item<S>) -> Self {
        item.tensor
    }
}

impl<B: Backend, const D: usize> Record for Tensor<B, D, Int> {
    type Item<S: RecordSettings> = IntTensorSerde<B, D, S>;

    fn into_item<S: RecordSettings>(self) -> Self::Item<S> {
        IntTensorSerde::new(self)
    }

    fn from_item<S: RecordSettings>(item: Self::Item<S>) -> Self {
        item.tensor
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
