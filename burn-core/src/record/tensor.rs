use super::{PrecisionSettings, Record};
use burn_tensor::{backend::Backend, Bool, DataSerialize, Int, Tensor};
use serde::{Deserialize, Serialize};

/// This struct implements serde to lazily serialize and deserialize a float tensor
/// using the given [record settings](RecordSettings).
#[derive(new, Clone, Debug)]
pub struct FloatTensorSerde<S: PrecisionSettings> {
    data: DataSerialize<S::FloatElem>,
}

/// This struct implements serde to lazily serialize and deserialize an int tensor
/// using the given [record settings](RecordSettings).
#[derive(new, Clone, Debug)]
pub struct IntTensorSerde<S: PrecisionSettings> {
    data: DataSerialize<S::IntElem>,
}

/// This struct implements serde to lazily serialize and deserialize an bool tensor.
#[derive(new, Clone, Debug)]
pub struct BoolTensorSerde {
    data: DataSerialize<bool>,
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
        let data = DataSerialize::<S::FloatElem>::deserialize(deserializer)?;

        Ok(Self::new(data))
    }
}

// #[cfg(not(target_family = "wasm"))]
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
        let data = DataSerialize::<S::IntElem>::deserialize(deserializer)?;
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
        let data = DataSerialize::<bool>::deserialize(deserializer)?;

        Ok(Self::new(data))
    }
}

// --- RECORD IMPLEMENTATIONS --- //

impl<B: Backend, const D: usize> Record for Tensor<B, D> {
    type Item<S: PrecisionSettings> = FloatTensorSerde<S>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        #[cfg(target_family = "wasm")]
        todo!("Recording float tensors isn't yet supported on wasm.");

        #[cfg(not(target_family = "wasm"))]
        FloatTensorSerde::new(self.into_data().convert().serialize())
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>) -> Self {
        Tensor::from_data(item.data.convert::<B::FloatElem>())
    }
}

impl<B: Backend, const D: usize> Record for Tensor<B, D, Int> {
    type Item<S: PrecisionSettings> = IntTensorSerde<S>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        #[cfg(target_family = "wasm")]
        todo!("Recording int tensors isn't yet supported on wasm.");

        #[cfg(not(target_family = "wasm"))]
        IntTensorSerde::new(self.into_data().convert().serialize())
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>) -> Self {
        Tensor::from_data(item.data.convert())
    }
}

impl<B: Backend, const D: usize> Record for Tensor<B, D, Bool> {
    type Item<S: PrecisionSettings> = BoolTensorSerde;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        #[cfg(target_family = "wasm")]
        todo!("Recording bool tensors isn't yet supported on wasm.");

        #[cfg(not(target_family = "wasm"))]
        BoolTensorSerde::new(self.into_data().serialize())
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>) -> Self {
        Tensor::from_data(item.data)
    }
}
