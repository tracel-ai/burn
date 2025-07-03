use std::collections::HashMap;

use burn::{
    module::{Module, ModuleVisitor},
    prelude::Backend,
    record::{FullPrecisionSettings, Record},
    tensor::{Tensor, TensorData},
};
use safetensors::SafeTensorError;
use serde::Serialize;

use super::serializer::Serializer;

struct SafetensorsTensorData(TensorData);

impl safetensors::View for SafetensorsTensorData {
    fn dtype(&self) -> safetensors::Dtype {
        match self.0.dtype {
            burn::tensor::DType::F64 => safetensors::Dtype::F64,
            burn::tensor::DType::F32 => safetensors::Dtype::F32,
            burn::tensor::DType::Flex32 => unimplemented!(), // TODO: check mapping
            burn::tensor::DType::F16 => safetensors::Dtype::F16,
            burn::tensor::DType::BF16 => safetensors::Dtype::BF16,
            burn::tensor::DType::I64 => safetensors::Dtype::I64,
            burn::tensor::DType::I32 => safetensors::Dtype::I32,
            burn::tensor::DType::I16 => safetensors::Dtype::I16,
            burn::tensor::DType::I8 => safetensors::Dtype::I8,
            burn::tensor::DType::U64 => safetensors::Dtype::U64,
            burn::tensor::DType::U32 => safetensors::Dtype::U32,
            burn::tensor::DType::U16 => safetensors::Dtype::U16,
            burn::tensor::DType::U8 => safetensors::Dtype::U8,
            burn::tensor::DType::Bool => safetensors::Dtype::BOOL,
            burn::tensor::DType::QFloat(_quant_scheme) => unimplemented!(),
        }
    }

    fn shape(&self) -> &[usize] {
        &self.0.shape
    }

    fn data(&self) -> std::borrow::Cow<[u8]> {
        std::borrow::Cow::Borrowed(&self.0.bytes)
    }

    fn data_len(&self) -> usize {
        self.0.bytes.len()
    }
}

struct SafetensorsModuleVisitor {
    tensors: HashMap<String, SafetensorsTensorData>,
}

impl<B: Backend> ModuleVisitor<B> for SafetensorsModuleVisitor {
    fn visit_float<const D: usize>(&mut self, id: burn::module::ParamId, tensor: &Tensor<B, D>) {
        self.tensors
            .insert(id.to_string(), SafetensorsTensorData(tensor.to_data()));
    }

    fn visit_int<const D: usize>(
        &mut self,
        id: burn::module::ParamId,
        tensor: &Tensor<B, D, burn::prelude::Int>,
    ) {
        self.tensors
            .insert(id.to_string(), SafetensorsTensorData(tensor.to_data()));
    }

    fn visit_bool<const D: usize>(
        &mut self,
        id: burn::module::ParamId,
        tensor: &Tensor<B, D, burn::prelude::Bool>,
    ) {
        self.tensors
            .insert(id.to_string(), SafetensorsTensorData(tensor.to_data()));
    }
}

pub fn to_safetensors<B: Backend, T: Module<B>>(module: T) -> Result<Vec<u8>, SafeTensorError> {
    let ser = Serializer::new();
    let serialized = module
        .clone()
        .into_record()
        .into_item::<FullPrecisionSettings>()
        .serialize(ser)
        .unwrap();
    let name2tensor_id = serialized.flatten("".into());
    let name2tensor_id: HashMap<String, String> = name2tensor_id
        .into_iter()
        // filter out all fields that do not finish with ".id"
        .filter(|(key, _)| key.chars().rev().take(3).collect::<String>() == "di.")
        // remove the last three characters ".id" from "tensor.name.id"
        .map(|(key, name)| ((key[..key.len() - 3]).to_string(), name))
        .collect();
    let mut visitor = SafetensorsModuleVisitor {
        tensors: HashMap::new(),
    };
    module.visit(&mut visitor);

    let name2tensor = name2tensor_id.into_iter().map(|(name, id)| {
        (
            name,
            visitor
                .tensors
                .remove(&id)
                .expect("the id must correspond to a tensor parameter in the module"),
        )
    });
    safetensors::serialize(name2tensor, None)
}
