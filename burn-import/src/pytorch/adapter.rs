use burn::{
    module::Param,
    record::{PrecisionSettings, Record},
    tensor::{backend::Backend, Tensor},
};
use burn_ndarray::NdArray;
use serde::Serialize;

use crate::record::{adapter::BurnModuleAdapter, reader::NestedValue, ser::Serializer};

type DummyBackend = NdArray<f32>;

pub struct PyTorchAdapter<PS: PrecisionSettings> {
    _precision_settings: std::marker::PhantomData<PS>,
}

impl<PS: PrecisionSettings> BurnModuleAdapter for PyTorchAdapter<PS> {
    fn adapt_linear(data: NestedValue) -> NestedValue {
        let mut map = data.get_map().unwrap().clone();

        let weight = map.remove("weight").unwrap();

        let weight: Param<Tensor<DummyBackend, 2>> = weight.de_into::<_, PS>().unwrap();

        let weight_transposed = Param::from(weight.val().transpose());

        map.insert(
            "weight".to_string(),
            serialize::<PS, _, 2>(weight_transposed),
        );

        NestedValue::Map(map)
    }
}

/// Helper function to serialize a param tensor.
fn serialize<PS, B, const D: usize>(val: Param<Tensor<B, D>>) -> NestedValue
where
    B: Backend,
    PS: PrecisionSettings,
{
    let serializer = Serializer::new();

    val.into_item::<PS>().serialize(serializer).unwrap()
}
