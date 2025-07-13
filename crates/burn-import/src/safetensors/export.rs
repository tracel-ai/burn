use burn::{
    module::Module,
    prelude::Backend,
    record::{FullPrecisionSettings, Record},
};
use safetensors::SafeTensorError;
use serde::Serialize;

use super::serializer::FlattenSerializer;

/// Generates the safetensors serialization of a [Module]
pub fn to_safetensors<B: Backend, T: Module<B>>(module: T) -> Result<Vec<u8>, SafeTensorError> {
    let mut ser = FlattenSerializer::new();
    module
        .into_record()
        .into_item::<FullPrecisionSettings>()
        .serialize(&mut ser)
        .unwrap();
    safetensors::serialize(ser.into_map(), None)
}
