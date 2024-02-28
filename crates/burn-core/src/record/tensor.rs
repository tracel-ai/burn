use super::{PrecisionSettings, Record};
use burn_tensor::{backend::Backend, Bool, DynData, DynRankData, DynTensor, Int, Tensor};
use serde::{Deserialize, Serialize};

pub enum DynTensorSerde<S: PrecisionSettings> {
    Float(FloatTensorSerde<S>),
    Int(IntTensorSerde<S>),
    Bool(BoolTensorSerde),
}

impl<S: PrecisionSettings> From<DynData> for DynTensorSerde<S> {
    fn from(value: DynData) -> Self {
        match value {
            DynData::Float(data) => Self::Float(FloatTensorSerde::new(data.convert())),
            DynData::Int(data) => Self::Int(IntTensorSerde::new(data.convert())),
            DynData::Bool(data) => Self::Bool(BoolTensorSerde::new(data)),
        }
    }
}

impl<S: PrecisionSettings> From<DynTensorSerde<S>> for DynData {
    fn from(value: DynTensorSerde<S>) -> Self {
        match value {
            DynTensorSerde::Float(tensor) => Self::Float(tensor.data.convert()),
            DynTensorSerde::Int(tensor) => Self::Int(tensor.data.convert()),
            DynTensorSerde::Bool(tensor) => Self::Bool(tensor.data),
        }
    }
}

/// This struct implements serde to lazily serialize and deserialize a float tensor
/// using the given [record settings](RecordSettings).
#[derive(new, Clone, Debug)]
pub struct FloatTensorSerde<S: PrecisionSettings> {
    data: DynRankData<S::FloatElem>,
}

/// This struct implements serde to lazily serialize and deserialize an int tensor
/// using the given [record settings](RecordSettings).
#[derive(new, Clone, Debug)]
pub struct IntTensorSerde<S: PrecisionSettings> {
    data: DynRankData<S::IntElem>,
}

/// This struct implements serde to lazily serialize and deserialize an bool tensor.
#[derive(new, Clone, Debug)]
pub struct BoolTensorSerde {
    data: DynRankData<bool>,
}

// --- SERDE IMPLEMENTATIONS --- //

impl<S: PrecisionSettings> Serialize for FloatTensorSerde<S> {
    fn serialize<Se>(&self, serializer: Se) -> Result<Se::Ok, Se::Error>
    where
        Se: serde::Serializer,
    {
        self.data.serialize(serializer)
    }
}

impl<'de, S: PrecisionSettings> Deserialize<'de> for FloatTensorSerde<S> {
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: serde::Deserializer<'de>,
    {
        let data = DynRankData::<S::FloatElem>::deserialize(deserializer)?;

        Ok(Self::new(data))
    }
}

impl<S: PrecisionSettings> Serialize for IntTensorSerde<S> {
    fn serialize<Se>(&self, serializer: Se) -> Result<Se::Ok, Se::Error>
    where
        Se: serde::Serializer,
    {
        self.data.serialize(serializer)
    }
}

impl<'de, S: PrecisionSettings> Deserialize<'de> for IntTensorSerde<S> {
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: serde::Deserializer<'de>,
    {
        let data = DynRankData::<S::IntElem>::deserialize(deserializer)?;
        Ok(Self::new(data))
    }
}

impl Serialize for BoolTensorSerde {
    fn serialize<Se>(&self, serializer: Se) -> Result<Se::Ok, Se::Error>
    where
        Se: serde::Serializer,
    {
        self.data.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for BoolTensorSerde {
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: serde::Deserializer<'de>,
    {
        let data = DynRankData::<bool>::deserialize(deserializer)?;

        Ok(Self::new(data))
    }
}

// --- RECORD IMPLEMENTATIONS --- //

impl<B: Backend, const D: usize> Record<B> for Tensor<B, D> {
    type Item<S: PrecisionSettings> = FloatTensorSerde<S>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        #[cfg(all(not(feature = "wasm-sync"), target_family = "wasm"))]
        todo!("Recording float tensors isn't yet supported on wasm.");

        #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
        FloatTensorSerde::new(self.into_data().convert().serialize())
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        Tensor::from_data(item.data.convert::<B::FloatElem>(), device)
    }
}

impl<B: Backend, const D: usize> Record<B> for Tensor<B, D, Int> {
    type Item<S: PrecisionSettings> = IntTensorSerde<S>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        #[cfg(all(not(feature = "wasm-sync"), target_family = "wasm"))]
        todo!("Recording int tensors isn't yet supported on wasm.");

        #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
        IntTensorSerde::new(self.into_data().convert().serialize())
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        Tensor::from_data(item.data.convert(), device)
    }
}

impl<B: Backend, const D: usize> Record<B> for Tensor<B, D, Bool> {
    type Item<S: PrecisionSettings> = BoolTensorSerde;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        #[cfg(all(not(feature = "wasm-sync"), target_family = "wasm"))]
        todo!("Recording bool tensors isn't yet supported on wasm.");

        #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
        BoolTensorSerde::new(self.into_data().serialize())
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        Tensor::from_data(item.data, device)
    }
}

impl<B: Backend> Record<B> for DynTensor<B> {
    type Item<S: PrecisionSettings> = DynTensorSerde<S>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        #[cfg(all(not(feature = "wasm-sync"), target_family = "wasm"))]
        todo!("Recording dynamic tensors isn't yet supported on wasm.");

        #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
        self.into_data().into()
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        Tensor::from_data(item.into(), device)
    }
}
