use std::path::Path;
use std::str::{FromStr, from_utf8};

use crate::external_data::{ExternalDataInfo, is_external_data};

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
    ExternalDataError(String),
}

/// Convert TensorProto to TensorData with support for external data
///
/// # Arguments
/// * `tensor` - The TensorProto to convert
/// * `base_dir` - Base directory for resolving external data file paths (optional)
pub fn tensor_proto_to_data(
    tensor: TensorProto,
    base_dir: Option<&Path>,
) -> Result<TensorData, ParseError> {
    // Check if tensor uses external data storage
    if is_external_data(tensor.data_location) {
        // Parse external data information
        let external_info =
            ExternalDataInfo::from_proto(&tensor.external_data).ok_or_else(|| {
                ParseError::ExternalDataError(
                    "Tensor marked as external but no external_data information found".to_string(),
                )
            })?;

        // Base directory is required for external data
        let base_dir = base_dir.ok_or_else(|| {
            ParseError::ExternalDataError(
                "Base directory required for loading external data".to_string(),
            )
        })?;

        // Read external data from file
        let raw_data = external_info.read_data(base_dir).map_err(|e| {
            ParseError::ExternalDataError(format!("Failed to read external data: {}", e))
        })?;

        // Convert raw bytes to typed data using the same logic as raw_data handling below
        let shape = convert_shape(tensor.dims);
        let elem = element_type_from_proto(tensor.data_type)
            .map_err(|e| ParseError::ExternalDataError(format!("Invalid data type: {}", e)))?;

        let data = match elem {
            ElementType::Float32 => Data::Float32s(cast_vec_with_fallback(raw_data)),
            ElementType::Float64 => Data::Float64s(cast_vec_with_fallback(raw_data)),
            ElementType::Float16 => Data::Float16s(cast_vec_with_fallback(raw_data)),
            ElementType::Int32 => Data::Int32s(cast_vec_with_fallback(raw_data)),
            ElementType::Int64 => Data::Int64s(cast_vec_with_fallback(raw_data)),
            ElementType::Uint16 => Data::Uint16s(cast_vec_with_fallback(raw_data)),
            ElementType::Uint8 => Data::Uint8s(raw_data),
            ElementType::Int8 => Data::Int8s(raw_data.into_iter().map(|b| b as i8).collect()),
            ElementType::Bool => Data::Bools(raw_data.into_iter().map(|b| b != 0).collect()),
            ElementType::String => {
                return Err(ParseError::ExternalDataError(
                    "String tensors not supported in external data".into(),
                ));
            }
        };

        return Ok(TensorData { shape, data });
    }

    // Fallback to standard TryFrom conversion for non-external data
    TensorData::try_from(tensor)
}

/// Convert a vector of AttributeProto to a HashMap of AttributeValue
impl TryFrom<TensorProto> for TensorData {
    type Error = ParseError;

    fn try_from(tensor: TensorProto) -> Result<TensorData, Self::Error> {
        // When using TryFrom, we don't have access to base_dir, so external data is not supported
        // This is mainly for backward compatibility and attribute tensors
        tensor_proto_to_data(tensor, None)
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
