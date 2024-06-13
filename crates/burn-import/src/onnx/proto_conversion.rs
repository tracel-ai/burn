use std::str::{from_utf8, FromStr};

use crate::onnx::ir::TensorType;

use super::from_onnx::GraphData;
use super::ir::Dim;
use super::ir::{
    ArgType, Argument, AttributeValue, Attributes, Data, ElementType, Node, NodeType, Tensor,
};
use super::protos::{
    attribute_proto::AttributeType, tensor_proto::DataType, tensor_shape_proto::dimension::Value,
    type_proto, AttributeProto, NodeProto, TensorProto, TensorShapeProto, ValueInfoProto,
};

use bytemuck::cast_slice;
use protobuf::Enum;

/// Error type for parsing ONNX model
#[derive(Debug)]
pub enum ParseError {
    VariantNotFound,
}

/// Convert a vector of AttributeProto to a HashMap of AttributeValue
impl TryFrom<TensorProto> for Tensor {
    type Error = ParseError;
    fn try_from(tensor: TensorProto) -> Result<Tensor, Self::Error> {
        let (elem_type, data) = match DataType::from_i32(tensor.data_type).unwrap() {
            DataType::FLOAT => (
                ElementType::Float32,
                // Convert the raw data to a vector of floats
                if !tensor.raw_data.is_empty() {
                    Data::Float32s(cast_slice(&tensor.raw_data[..]).to_vec())
                } else {
                    Data::Float32s(tensor.float_data)
                },
            ),
            DataType::INT16 => {
                // TODO : Add support for int16 by converting to int32
                todo!("Add support for int16");
            }
            DataType::INT32 => (
                ElementType::Int32,
                // Convert the raw data to a vector of ints
                if !tensor.raw_data.is_empty() {
                    Data::Int32s(cast_slice(&tensor.raw_data[..]).to_vec())
                } else {
                    Data::Int32s(tensor.int32_data)
                },
            ),
            DataType::INT64 => (
                ElementType::Int64,
                // Convert the raw data to a vector of ints
                if !tensor.raw_data.is_empty() {
                    Data::Int64s(cast_slice(&tensor.raw_data[..]).to_vec())
                } else {
                    Data::Int64s(tensor.int64_data)
                },
            ),
            DataType::DOUBLE => (
                ElementType::Float64,
                // Convert the raw data to a vector of floats
                if !tensor.raw_data.is_empty() {
                    Data::Float64s(cast_slice(&tensor.raw_data[..]).to_vec())
                } else {
                    Data::Float64s(tensor.double_data)
                },
            ),
            DataType::BOOL => (ElementType::Bool, {
                assert!(!tensor.raw_data.is_empty());
                Data::Bools(tensor.raw_data.iter().map(|x| *x != 0).collect())
            }),
            // TODO : Add more types
            _ => {
                return Err(ParseError::VariantNotFound);
            }
        };
        let shape = convert_shape(tensor.dims);

        Ok(Tensor {
            elem_type,
            dim: shape.len(),
            shape: Some(shape),
            data: Some(data),
        })
    }
}

impl TryFrom<TensorShapeProto> for Vec<usize> {
    type Error = ParseError;
    fn try_from(shape: TensorShapeProto) -> Result<Vec<usize>, Self::Error> {
        let mut result = Vec::new();

        for dim in shape.dim {
            if let Value::DimValue(value) = dim.value.unwrap() {
                result.push(value as usize);
            }
        }

        Ok(result)
    }
}

/// Convert a vector of AttributeProto to a HashMap of AttributeValue
impl TryFrom<&type_proto::Tensor> for Tensor {
    type Error = ParseError;
    fn try_from(tensor: &type_proto::Tensor) -> Result<Tensor, Self::Error> {
        let elem_type = match DataType::from_i32(tensor.elem_type).unwrap() {
            DataType::FLOAT => ElementType::Float32,
            DataType::INT32 => ElementType::Int32,
            DataType::INT64 => ElementType::Int64,
            DataType::DOUBLE => ElementType::Float64,
            DataType::BOOL => ElementType::Bool,

            // TODO : Add more types
            _ => {
                return Err(ParseError::VariantNotFound);
            }
        };

        let shape_proto = tensor.shape.clone().unwrap();
        let shape: Vec<usize> = shape_proto.try_into().unwrap();

        Ok(Tensor {
            elem_type,
            dim: shape.len(),
            shape: Some(shape),
            data: None,
        })
    }
}

fn convert_vec_tensor_proto(tensors: Vec<TensorProto>) -> Result<Vec<Tensor>, ParseError> {
    let mut result = Vec::new();
    for tensor in tensors {
        result.push(Tensor::try_from(tensor)?);
    }
    Ok(result)
}

/// Convert a vector of AttributeProto to a HashMap of AttributeValue
impl TryFrom<AttributeProto> for AttributeValue {
    type Error = ParseError;

    fn try_from(attr: AttributeProto) -> Result<AttributeValue, Self::Error> {
        let value = match attr.type_.unwrap() {
            AttributeType::FLOAT => AttributeValue::Float32(attr.f),
            AttributeType::INT => AttributeValue::Int64(attr.i),
            AttributeType::STRING => AttributeValue::String(to_string(attr.s)),

            // warning: tensor can be empty TODO: check if it is empty
            AttributeType::TENSOR => AttributeValue::Tensor(Tensor::try_from(attr.t.unwrap())?),

            // Graph is not supported for now
            // AttributeType::GRAPH => AttributeValue::Graph(attr.g),
            AttributeType::FLOATS => AttributeValue::Float32s(attr.floats),
            AttributeType::INTS => AttributeValue::Int64s(attr.ints),
            AttributeType::STRINGS => AttributeValue::Strings(to_string_vec(attr.strings)),
            AttributeType::TENSORS => {
                AttributeValue::Tensors(convert_vec_tensor_proto(attr.tensors)?)
            }
            // AttributeType::GRAPHS => AttributeValue::Graphs(attr.graphs),
            // AttributeType::SPARSE_TENSORS => AttributeValue::SparseTensors(attr.sparse_tensors),
            // AttributeType::SPARSE_TENSOR => AttributeValue::SparseTensor(attr.sparse_tensor),
            _ => {
                return Err(ParseError::VariantNotFound);
            }
        };

        Ok(value)
    }
}

/// Convert a vector of AttributeProto to a HashMap of AttributeValue
pub fn convert_vec_attrs_proto(attrs: Vec<AttributeProto>) -> Attributes {
    let mut result = Attributes::new();
    for attr in attrs {
        result.insert(attr.name.clone(), AttributeValue::try_from(attr).unwrap());
    }
    result
}

pub fn convert_node_proto(node: &NodeProto, graph_data: &GraphData) -> Node {
    let name = node.name.clone();

    log::debug!("Converting ONNX node with type {:?}", node.op_type.as_str());

    let inputs = node.input.iter().map(|x| graph_data.init_in(x)).collect();

    let outputs = node
        .output
        .iter()
        .map(|x| Argument::new(x.to_string()))
        .collect();

    let attrs = convert_vec_attrs_proto(node.attribute.clone());

    let node_type = NodeType::from_str(node.op_type.as_str()).expect("Unknown node type");

    Node {
        node_type,
        name,
        inputs,
        outputs,
        attrs,
    }
}

fn to_string(bytes: Vec<u8>) -> String {
    from_utf8(bytes.as_slice()).unwrap().to_string()
}

fn to_string_vec(bytes: Vec<Vec<u8>>) -> Vec<String> {
    bytes.iter().map(|b| to_string(b.clone())).collect()
}

fn convert_shape(shape: Vec<i64>) -> Vec<usize> {
    shape.iter().map(|s| *s as usize).collect()
}

impl TryFrom<ValueInfoProto> for Argument {
    type Error = ParseError;

    fn try_from(value: ValueInfoProto) -> Result<Argument, Self::Error> {
        let name = value.name.clone();
        let proto_type = value.type_.unwrap();

        if !proto_type.has_tensor_type() {
            panic!("Unsupported argument type {:?}", proto_type);
        }

        let tensor_proto = proto_type.tensor_type();

        let elem_type = match DataType::from_i32(tensor_proto.elem_type).unwrap() {
            DataType::FLOAT => ElementType::Float32,
            DataType::INT32 => ElementType::Int32,
            DataType::INT64 => ElementType::Int64,
            DataType::DOUBLE => ElementType::Float64,
            DataType::BOOL => ElementType::Bool,
            _ => {
                return Err(ParseError::VariantNotFound);
            }
        };

        let tensor_type = TensorType {
            dim: tensor_proto.shape.dim.len(),
            elem_type,
            shape: Some(
                tensor_proto
                    .shape
                    .dim
                    .iter()
                    .map(|x| x.dim_value() as Dim)
                    .collect(),
            ),
        };

        let ty = ArgType::Tensor(tensor_type);

        Ok(Argument {
            ty,
            name,
            value: None,
            passed: false,
        })
    }
}
