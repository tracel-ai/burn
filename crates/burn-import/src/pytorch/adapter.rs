use burn::{
    module::Param,
    record::{PrecisionSettings, Record},
    tensor::{backend::Backend, Tensor},
};

use burn::record::serde::{
    adapter::{BurnModuleAdapter, DefaultAdapter},
    data::NestedValue,
    ser::Serializer,
};

use serde::Serialize;

/// A PyTorch adapter for the Burn module used during deserialization.
///
/// Not all Burn module correspond to a PyTorch module. Therefore,
/// we need to adapt the Burn module to a PyTorch module. We implement
/// only those that differ.
pub struct PyTorchAdapter<PS: PrecisionSettings, B: Backend> {
    _precision_settings: std::marker::PhantomData<(PS, B)>,
}

impl<PS: PrecisionSettings, B: Backend> BurnModuleAdapter for PyTorchAdapter<PS, B> {
    fn adapt_linear(data: NestedValue) -> NestedValue {
        // Get the current module in the form of map.
        let mut map = data.as_map().expect("Failed to get map from NestedValue");

        // Get/remove the weight parameter.
        let weight = map
            .remove("weight")
            .expect("Failed to find 'weight' key in map");

        // Convert the weight parameter to a tensor (use default device, since it's quick operation).
        let weight: Param<Tensor<B, 2>> = weight
            .try_into_record::<_, PS, DefaultAdapter, B>(&B::Device::default())
            .expect("Failed to deserialize weight");

        // Do not capture transpose op when using autodiff backend
        let weight = weight.set_require_grad(false);
        // Transpose the weight tensor.
        let weight_transposed = Param::from_tensor(weight.val().transpose());

        // Insert the transposed weight tensor back into the map.
        map.insert(
            "weight".to_owned(),
            serialize::<PS, _, 2>(weight_transposed),
        );

        // Return the modified map.
        NestedValue::Map(map)
    }

    fn adapt_group_norm(data: NestedValue) -> NestedValue {
        rename_weight_bias(data)
    }

    fn adapt_batch_norm(data: NestedValue) -> NestedValue {
        rename_weight_bias(data)
    }

    fn adapt_layer_norm(data: NestedValue) -> NestedValue {
        rename_weight_bias(data)
    }
}

/// Helper function to serialize a param tensor.
fn serialize<PS, B, const D: usize>(val: Param<Tensor<B, D>>) -> NestedValue
where
    B: Backend,
    PS: PrecisionSettings,
{
    let serializer = Serializer::new();

    val.into_item::<PS>()
        .serialize(serializer)
        .expect("Failed to serialize the item")
}

/// Helper function to rename the weight and bias parameters to gamma and beta.
///
/// This is needed because PyTorch uses different names for the normalizer parameter
/// than Burn. Burn uses gamma and beta, while PyTorch uses weight and bias.
fn rename_weight_bias(data: NestedValue) -> NestedValue {
    // Get the current module in the form of map.
    let mut map = data.as_map().expect("Failed to get map from NestedValue");

    // Rename the weight parameter to gamma.
    let weight = map
        .remove("weight")
        .expect("Failed to find 'weight' key in map");

    map.insert("gamma".to_owned(), weight);

    // Rename the bias parameter to beta.
    let bias = map
        .remove("bias")
        .expect("Failed to find 'bias' key in map");

    map.insert("beta".to_owned(), bias);

    // Return the modified map.
    NestedValue::Map(map)
}
