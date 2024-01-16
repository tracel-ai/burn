use crate::record::{adapter::{BurnModuleAdapter, DefaultAdapter}, data::NestedValue, ser::Serializer};

use burn::{
    module::Param,
    record::{PrecisionSettings, Record},
    tensor::{backend::Backend, Tensor},
};

use burn_ndarray::NdArray;

use serde::Serialize;

/// A dummy backend used to transform tensors.
type DummyBackend = NdArray<f32>;

/// A PyTorch adapter for the Burn module used during deserialization.
///
/// Not all Burn module correspond to a PyTorch module. Therefore,
/// we need to adapt the Burn module to a PyTorch module. We implement
/// only those that differ.
pub struct PyTorchAdapter<PS: PrecisionSettings> {
    _precision_settings: std::marker::PhantomData<PS>,
}

impl<PS: PrecisionSettings> BurnModuleAdapter for PyTorchAdapter<PS> {
    fn adapt_linear(data: NestedValue) -> NestedValue {
        // Get the current module in the form of map.
        let mut map = data.get_map().unwrap().clone();

        // Get/remove the weight parameter.
        let weight = map.remove("weight").unwrap();

        // Convert the weight parameter to a tensor.
        let weight: Param<Tensor<DummyBackend, 2>> =
            weight.de_into::<_, PS, DefaultAdapter>().unwrap();

        // Transpose the weight tensor.
        let weight_transposed = Param::from(weight.val().transpose());

        // Insert the transposed weight tensor back into the map.
        map.insert(
            "weight".to_string(),
            serialize::<PS, _, 2>(weight_transposed),
        );

        // Return the modified map.
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
