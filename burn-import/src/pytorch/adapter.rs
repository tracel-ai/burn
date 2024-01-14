use burn::{
    module::Param,
    record::{FullPrecisionSettings, Record},
    tensor::Tensor,
};
use burn_ndarray::NdArray;
use serde::Serialize;

use super::{reader::NestedValue, ser::Serializer};

pub trait BurnModuleAdapter: Sized {
    fn adapt(name: &str, data: NestedValue) -> NestedValue {
        match name {
            "BatchNorm" => Self::adapt_batch_norm(data),
            "Conv1d" => Self::adapt_conv1d(data),
            "Conv2d" => Self::adapt_conv2d(data),
            "ConvTranspose1d" => Self::adapt_conv_transpose_1d(data),
            "ConvTranspose2d" => Self::adapt_conv_transpose_2d(data),
            "Embedding" => Self::adapt_embedding(data),
            "GroupNorm" => Self::adapt_group_norm(data),
            "LayerNorm" => Self::adapt_layer_norm(data),
            "Linear" => Self::adapt_linear(data),
            _ => data,
        }
    }

    fn adapt_linear(data: NestedValue) -> NestedValue {
        data
    }

    fn adapt_conv1d(data: NestedValue) -> NestedValue {
        data
    }

    fn adapt_conv2d(data: NestedValue) -> NestedValue {
        data
    }

    fn adapt_conv_transpose_1d(data: NestedValue) -> NestedValue {
        data
    }

    fn adapt_conv_transpose_2d(data: NestedValue) -> NestedValue {
        data
    }

    fn adapt_embedding(data: NestedValue) -> NestedValue {
        data
    }

    fn adapt_group_norm(data: NestedValue) -> NestedValue {
        data
    }

    fn adapt_layer_norm(data: NestedValue) -> NestedValue {
        data
    }

    fn adapt_batch_norm(data: NestedValue) -> NestedValue {
        data
    }
}

/// Default adapter that takes no action.
pub struct DefaultAdapter;
impl BurnModuleAdapter for DefaultAdapter {}

type B = NdArray<f32>;

pub struct PyTorchAdapter;

impl BurnModuleAdapter for PyTorchAdapter {
    fn adapt_linear(data: NestedValue) -> NestedValue {
        let mut map = data.get_map().unwrap().clone();

        let weight = map.remove("weight").unwrap();

        let weight: Param<Tensor<B, 2>> = weight.de_into::<_, FullPrecisionSettings>().unwrap();

        let weight_transposed = Param::from(weight.val().transpose());

        map.insert("weight".to_string(), weight_transposed.into());

        NestedValue::Map(map)
    }
}

impl From<Param<Tensor<B, 2>>> for NestedValue {
    fn from(val: Param<Tensor<B, 2>>) -> Self {
        let serializer = Serializer::new();

        val.into_item::<FullPrecisionSettings>()
            .serialize(serializer)
            .unwrap()
    }
}
