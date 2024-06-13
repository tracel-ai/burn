use core::marker::PhantomData;

use super::{PrecisionSettings, Record};
use burn_tensor::{backend::Backend, Bool, DataSerialize, Int, Tensor, TensorData};
use serde::{Deserialize, Serialize};

/// Data deserialization with backward compatibility.
#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum DataDeserialize<const D: usize, E> {
    V1(DataSerialize<E>),
    V2(TensorData<D>),
}

/// This struct implements serde to lazily serialize and deserialize a float tensor
/// using the given [record settings](RecordSettings).
#[derive(new, Clone, Debug)]
pub struct FloatTensorSerde<const D: usize, S: PrecisionSettings> {
    data: TensorData<D>,
    _e: PhantomData<S::FloatElem>,
}

/// This struct implements serde to lazily serialize and deserialize an int tensor
/// using the given [record settings](RecordSettings).
#[derive(new, Clone, Debug)]
pub struct IntTensorSerde<const D: usize, S: PrecisionSettings> {
    data: TensorData<D>,
    _e: PhantomData<S::IntElem>,
}

/// This struct implements serde to lazily serialize and deserialize an bool tensor.
#[derive(new, Clone, Debug)]
pub struct BoolTensorSerde<const D: usize> {
    data: TensorData<D>,
}

// --- SERDE IMPLEMENTATIONS --- //

impl<const D: usize, S: PrecisionSettings> Serialize for FloatTensorSerde<D, S> {
    fn serialize<Se>(&self, serializer: Se) -> Result<Se::Ok, Se::Error>
    where
        Se: serde::Serializer,
    {
        self.data.serialize(serializer)
    }
}

impl<'de, const D: usize, S: PrecisionSettings> Deserialize<'de> for FloatTensorSerde<D, S> {
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: serde::Deserializer<'de>,
    {
        let data = match DataDeserialize::<D, S::FloatElem>::deserialize(deserializer)? {
            DataDeserialize::V1(data) => data.into_tensor_data(),
            // NOTE: loading f32 weights with f16 precision will deserialize the f32 weights (bytes) first and then convert to f16
            DataDeserialize::V2(data) => data.convert::<S::FloatElem>(),
        };

        Ok(Self::new(data))
    }
}

impl<const D: usize, S: PrecisionSettings> Serialize for IntTensorSerde<D, S> {
    fn serialize<Se>(&self, serializer: Se) -> Result<Se::Ok, Se::Error>
    where
        Se: serde::Serializer,
    {
        self.data.serialize(serializer)
    }
}

impl<'de, const D: usize, S: PrecisionSettings> Deserialize<'de> for IntTensorSerde<D, S> {
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: serde::Deserializer<'de>,
    {
        let data = match DataDeserialize::<D, S::IntElem>::deserialize(deserializer)? {
            DataDeserialize::V1(data) => data.into_tensor_data(),
            DataDeserialize::V2(data) => data,
        };

        Ok(Self::new(data))
    }
}

impl<const D: usize> Serialize for BoolTensorSerde<D> {
    fn serialize<Se>(&self, serializer: Se) -> Result<Se::Ok, Se::Error>
    where
        Se: serde::Serializer,
    {
        self.data.serialize(serializer)
    }
}

impl<'de, const D: usize> Deserialize<'de> for BoolTensorSerde<D> {
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: serde::Deserializer<'de>,
    {
        let data = match DataDeserialize::<D, bool>::deserialize(deserializer)? {
            DataDeserialize::V1(data) => data.into_tensor_data(),
            DataDeserialize::V2(data) => data.convert::<bool>(),
        };

        Ok(Self::new(data))
    }
}

// --- RECORD IMPLEMENTATIONS --- //

impl<B: Backend, const D: usize> Record<B> for Tensor<B, D> {
    type Item<S: PrecisionSettings> = FloatTensorSerde<D, S>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        #[cfg(all(not(feature = "wasm-sync"), target_family = "wasm"))]
        todo!("Recording float tensors isn't yet supported on wasm.");

        #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
        FloatTensorSerde::new(self.into_data().convert::<S::FloatElem>())
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        Tensor::from_data(item.data.convert::<B::FloatElem>(), device)
    }
}

#[allow(deprecated)]
impl<B: Backend, const D: usize> Record<B> for Tensor<B, D, Int> {
    type Item<S: PrecisionSettings> = IntTensorSerde<D, S>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        #[cfg(all(not(feature = "wasm-sync"), target_family = "wasm"))]
        todo!("Recording int tensors isn't yet supported on wasm.");

        #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
        IntTensorSerde::new(self.into_data().convert::<S::IntElem>())
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        Tensor::from_data(item.data.convert::<B::IntElem>(), device)
    }
}

#[allow(deprecated)]
impl<B: Backend, const D: usize> Record<B> for Tensor<B, D, Bool> {
    type Item<S: PrecisionSettings> = BoolTensorSerde<D>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        #[cfg(all(not(feature = "wasm-sync"), target_family = "wasm"))]
        todo!("Recording bool tensors isn't yet supported on wasm.");

        #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
        BoolTensorSerde::new(self.into_data())
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        Tensor::from_data(item.data, device)
    }
}
