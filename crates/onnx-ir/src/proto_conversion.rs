use std::str::{FromStr, from_utf8};

use super::from_onnx::GraphData;
use super::from_onnx::element_type_from_proto;
use super::ir::{
    ArgType, Argument, AttributeValue, Attributes, Data, ElementType, Node, NodeType, TensorData,
};
use super::protos::{
    AttributeProto, NodeProto, TensorProto, TensorShapeProto, ValueInfoProto,
    attribute_proto::AttributeType, tensor_shape_proto::dimension::Value,
};
use crate::ir::TensorType;

use bytemuck::{cast_slice, try_cast_vec};

fn cast_vec_with_fallback<E: bytemuck::Pod>(raw_data: Vec<u8>) -> Vec<E> {
    // Zero-copy `try_cast_vec` with fallback when alignment and size are not compatible
    try_cast_vec(raw_data).unwrap_or_else(|(_e, raw_data)| cast_slice(&raw_data).to_vec())
}

/// Error type for parsing ONNX model
#[derive(Debug)]
pub enum ParseError {
    VariantNotFound(String),
}

/// Convert a vector of AttributeProto to a HashMap of AttributeValue
impl TryFrom<TensorProto> for TensorData {
    type Error = ParseError;

    fn try_from(tensor: TensorProto) -> Result<TensorData, Self::Error> {
        let shape = convert_shape(tensor.dims);
        let elem =
            element_type_from_proto(tensor.data_type).map_err(ParseError::VariantNotFound)?;

        let data = if !tensor.raw_data.is_empty() {
            match elem {
                ElementType::Float32 => Data::Float32s(cast_vec_with_fallback(tensor.raw_data)),
                ElementType::Float64 => Data::Float64s(cast_vec_with_fallback(tensor.raw_data)),
                ElementType::Float16 => Data::Float16s(cast_vec_with_fallback(tensor.raw_data)),
                ElementType::Int32 => Data::Int32s(cast_vec_with_fallback(tensor.raw_data)),
                ElementType::Int64 => Data::Int64s(cast_vec_with_fallback(tensor.raw_data)),
                ElementType::Uint16 => Data::Uint16s(cast_vec_with_fallback(tensor.raw_data)),
                ElementType::Uint8 => Data::Uint8s(tensor.raw_data), // keep bytes
                ElementType::Int8 => {
                    Data::Int8s(tensor.raw_data.into_iter().map(|b| b as i8).collect())
                }
                ElementType::Bool => {
                    Data::Bools(tensor.raw_data.into_iter().map(|b| b != 0).collect())
                }
                ElementType::String => panic!("String initializers unsupported"),
            }
        } else {
            match elem {
                ElementType::Float32 if !tensor.float_data.is_empty() => {
                    Data::Float32s(tensor.float_data)
                }
                ElementType::Float64 if !tensor.double_data.is_empty() => {
                    Data::Float64s(tensor.double_data)
                }
                ElementType::Int32 if !tensor.int32_data.is_empty() => {
                    Data::Int32s(tensor.int32_data)
                }
                ElementType::Int64 if !tensor.int64_data.is_empty() => {
                    Data::Int64s(tensor.int64_data)
                }
                ElementType::Bool if !tensor.int32_data.is_empty() => {
                    Data::Bools(tensor.int32_data.into_iter().map(|x| x != 0).collect())
                }
                ElementType::Uint8 => {
                    // accept weird exporters that stuff zp as int32_data
                    if !tensor.int32_data.is_empty() {
                        Data::Uint8s(tensor.int32_data.into_iter().map(|x| x as u8).collect())
                    } else {
                        return Err(ParseError::VariantNotFound("no data for UINT8".into()));
                    }
                }
                ElementType::Int8 => {
                    if !tensor.int32_data.is_empty() {
                        Data::Int8s(tensor.int32_data.into_iter().map(|x| x as i8).collect())
                    } else {
                        return Err(ParseError::VariantNotFound("no data for INT8".into()));
                    }
                }
                _ => {
                    return Err(ParseError::VariantNotFound(format!(
                        "empty/unsupported payload for {:?}",
                        elem
                    )));
                }
            }
        };
        Ok(TensorData { shape, data })
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

fn convert_vec_tensor_proto(tensors: Vec<TensorProto>) -> Result<Vec<TensorData>, ParseError> {
    let mut result = Vec::new();
    for tensor in tensors {
        result.push(TensorData::try_from(tensor)?);
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
            AttributeType::TENSOR => AttributeValue::Tensor(TensorData::try_from(attr.t.unwrap())?),

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
            attribute_type => {
                return Err(ParseError::VariantNotFound(format!("{attribute_type:?}")));
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
        let proto_type = value
            .type_
            .as_ref()
            .ok_or(ParseError::VariantNotFound("missing type".into()))?;

        if !proto_type.has_tensor_type() {
            panic!("Unsupported argument type {proto_type:?}");
        }

        let tensor_proto = proto_type.tensor_type();
        let elem_type =
            element_type_from_proto(tensor_proto.elem_type).map_err(ParseError::VariantNotFound)?;

        let ty = if tensor_proto.shape.dim.is_empty() {
            ArgType::Scalar(elem_type)
        } else {
            let has_unknown_dim = tensor_proto.shape.dim.iter().any(|dim| match &dim.value {
                None | Some(Value::DimParam(_)) => true,
                Some(Value::DimValue(_)) => false,
            });

            let static_shape = if has_unknown_dim {
                None
            } else {
                let shape: Vec<usize> = tensor_proto
                    .shape
                    .dim
                    .iter()
                    .filter_map(|d| {
                        if let Some(Value::DimValue(v)) = &d.value {
                            Some(*v as usize)
                        } else {
                            None
                        }
                    })
                    .collect();
                Some(shape)
            };

            ArgType::Tensor(TensorType {
                rank: tensor_proto.shape.dim.len(),
                elem_type,
                static_shape,
            })
        };

        Ok(Argument {
            ty,
            name,
            value: None,
            passed: false,
        })
    }
}
