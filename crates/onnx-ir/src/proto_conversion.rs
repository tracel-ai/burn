use std::str::{FromStr, from_utf8};

use super::graph_state::GraphState;
use super::ir::{
    ArgType, Argument, AttributeValue, Attributes, Node, NodeType, TensorData, TensorDataExt,
    TensorType,
};
use super::protos::{
    AttributeProto, NodeProto, TensorProto, TensorShapeProto, ValueInfoProto,
    attribute_proto::AttributeType, tensor_proto::DataType as DT,
    tensor_shape_proto::dimension::Value,
};

use burn_tensor::DType;
use protobuf::Enum;

/// Minimum required ONNX opset version
pub const MIN_OPSET_VERSION: i64 = 16;

/// Error type for parsing ONNX model
#[derive(Debug)]
pub enum ParseError {
    VariantNotFound(String),
}

/// Convert ONNX protobuf DataType to DType
pub fn element_type_from_proto(dt_i32: i32) -> Result<DType, String> {
    match DT::from_i32(dt_i32).ok_or_else(|| format!("unknown dtype {}", dt_i32))? {
        DT::FLOAT => Ok(DType::F32),
        DT::DOUBLE => Ok(DType::F64),
        DT::FLOAT16 => Ok(DType::F16),
        DT::INT64 => Ok(DType::I64),
        DT::INT32 => Ok(DType::I32),
        DT::UINT16 => Ok(DType::U16),
        DT::UINT8 => Ok(DType::U8),
        DT::INT8 => Ok(DType::I8),
        DT::BOOL => Ok(DType::Bool),
        DT::STRING => Err("String tensors not supported".to_string()),
        other => Err(format!("unsupported dtype {:?}", other)),
    }
}

/// Create an Argument and TensorData from an ONNX initializer
///
/// Converts ONNX tensor initializers (weights, biases, etc.) into IR types.
/// Handles various ONNX encoding quirks including scalars and empty tensors.
///
/// Returns (Argument with type info, TensorData with actual values)
pub fn argument_from_initializer(initializer: &TensorProto) -> (Argument, TensorData) {
    use crate::ir::ValueSource;

    let name = initializer.name.clone();

    // 1) Canonical path first.
    match TensorData::try_from(initializer.clone()) {
        Ok(td) => {
            let arg = if td.shape.is_empty() {
                // rank-0 (scalar)
                Argument {
                    name,
                    ty: ArgType::Scalar(td.elem_type()),
                    data_id: None,
                    value_source: ValueSource::Constant, // Initializers are constants
                    value_store: None,
                }
            } else {
                Argument {
                    name,
                    ty: ArgType::Tensor(TensorType {
                        dtype: td.elem_type(),
                        rank: td.shape.len(),
                        static_shape: Some(td.shape.to_vec()),
                    }),
                    data_id: None,
                    value_source: ValueSource::Constant, // Initializers are constants
                    value_store: None,
                }
            };
            (arg, td)
        }
        Err(orig_err) => {
            // 2) Fallback handling for scalars & empty tensors, with precise diagnostics.
            let dims: Vec<i64> = initializer.dims.clone();
            if dims.iter().any(|&d| d < 0) {
                panic!(
                    "invalid tensor shape (negative dims) for initializer '{}': {:?}",
                    name, dims
                );
            }

            // Element count implied by dims (treat [] as scalar => 1).
            let dim_elems: usize = if dims.is_empty() {
                1
            } else {
                dims.iter().map(|&d| d as usize).product()
            };

            // Payload len across typed fields (best-effort).
            let payload_len = {
                let i32n = initializer.int32_data.len();
                let i64n = initializer.int64_data.len();
                let f32n = initializer.float_data.len();
                let f64n = initializer.double_data.len();
                let sn = initializer.string_data.len();
                let typed = *[i32n, i64n, f32n, f64n, sn].iter().max().unwrap_or(&0);
                if typed > 0 {
                    typed
                } else {
                    // raw_data fallback: many exporters put single scalars here
                    if !initializer.raw_data.is_empty() && dim_elems == 1 {
                        1
                    } else {
                        0
                    }
                }
            };

            // 2.a) Accept scalar encodings: [] or [1] with one element.
            let looks_scalar = dims.is_empty() || (dims.len() == 1 && dims[0] == 1);
            if looks_scalar && payload_len == 1 {
                let td = TensorData::try_from(initializer.clone()).unwrap_or_else(|_| {
                    panic!(
                        "failed to decode scalar initializer '{}': dims={:?}",
                        name, dims
                    )
                });
                let arg = Argument {
                    name,
                    ty: ArgType::Scalar(td.elem_type()),
                    data_id: None,
                    value_source: ValueSource::Constant, // Initializers are constants
                    value_store: None,
                };
                return (arg, td);
            }

            // 2.b) Accept EMPTY tensors: dim_elems == 0 with payload_len == 0.
            if dim_elems == 0 && payload_len == 0 && !dims.is_empty() {
                // Map ONNX data_type -> DType.
                // (Covers common types used in initializers; extend as needed.)
                let dtype = element_type_from_proto(initializer.data_type).unwrap_or_else(|e| {
                    panic!(
                        "unsupported empty-tensor data_type={} for '{}': {}",
                        initializer.data_type, name, e
                    )
                });

                // Build empty tensor using burn-tensor
                let shape_usize: Vec<usize> = dims.iter().map(|&d| d as usize).collect();

                let td = match dtype {
                    DType::F32 => TensorData::new(Vec::<f32>::new(), shape_usize.clone()),
                    DType::F64 => TensorData::new(Vec::<f64>::new(), shape_usize.clone()),
                    DType::F16 => TensorData::new(Vec::<half::f16>::new(), shape_usize.clone()),
                    DType::I32 => TensorData::new(Vec::<i32>::new(), shape_usize.clone()),
                    DType::I64 => TensorData::new(Vec::<i64>::new(), shape_usize.clone()),
                    DType::U16 => TensorData::new(Vec::<u16>::new(), shape_usize.clone()),
                    DType::U8 => TensorData::new(Vec::<u8>::new(), shape_usize.clone()),
                    DType::I8 => TensorData::new(Vec::<i8>::new(), shape_usize.clone()),
                    DType::Bool => TensorData::new(Vec::<bool>::new(), shape_usize.clone()),
                    _ => panic!("Unsupported dtype {:?} for empty tensor", dtype),
                };

                let arg = Argument {
                    name,
                    ty: ArgType::Tensor(TensorType {
                        dtype,
                        rank: shape_usize.len(),
                        static_shape: Some(shape_usize),
                    }),
                    data_id: None,
                    value_source: ValueSource::Constant, // Initializers are constants
                    value_store: None,
                };
                return (arg, td);
            }

            // Not scalar, not empty-tensor; fail with context.
            panic!(
                "invalid tensor '{}' (dims {:?} => {} elems) with payload {} elems; original error: {:?}",
                name, dims, dim_elems, payload_len, orig_err
            );
        }
    }
}

/// Convert a vector of AttributeProto to a HashMap of AttributeValue
impl TryFrom<TensorProto> for TensorData {
    type Error = ParseError;

    fn try_from(tensor: TensorProto) -> Result<TensorData, Self::Error> {
        let shape = convert_shape(tensor.dims);
        let elem =
            element_type_from_proto(tensor.data_type).map_err(ParseError::VariantNotFound)?;

        // Optimize using burn-tensor's from_bytes_vec for direct byte conversion
        if !tensor.raw_data.is_empty() {
            // Types that can use zero-copy or minimal-copy from raw bytes
            match elem {
                // These types can use from_bytes_vec directly (just reinterpret bytes)
                DType::F32
                | DType::F64
                | DType::F16
                | DType::I32
                | DType::I64
                | DType::U16
                | DType::U8 => {
                    // Use from_bytes_vec to avoid intermediate typed Vec allocation
                    Ok(burn_tensor::TensorData::from_bytes_vec(
                        tensor.raw_data,
                        shape,
                        elem,
                    ))
                }
                // These types need element-wise conversion
                DType::I8 => {
                    let data: Vec<i8> = tensor.raw_data.into_iter().map(|b| b as i8).collect();
                    Ok(TensorData::new(data, shape))
                }
                DType::Bool => {
                    let data: Vec<bool> = tensor.raw_data.into_iter().map(|b| b != 0).collect();
                    Ok(TensorData::new(data, shape))
                }
                _ => Err(ParseError::VariantNotFound(format!(
                    "Unsupported dtype {:?}",
                    elem
                ))),
            }
        } else {
            // Calculate expected number of elements from shape
            let expected_elems: usize = shape.iter().product();

            match elem {
                DType::F32 if !tensor.float_data.is_empty() => {
                    Ok(TensorData::new(tensor.float_data, shape))
                }
                DType::F32 if expected_elems == 0 => {
                    // Empty tensor with zero elements
                    Ok(TensorData::new(Vec::<f32>::new(), shape))
                }
                DType::F64 if !tensor.double_data.is_empty() => {
                    Ok(TensorData::new(tensor.double_data, shape))
                }
                DType::F64 if expected_elems == 0 => {
                    Ok(TensorData::new(Vec::<f64>::new(), shape))
                }
                DType::I32 if !tensor.int32_data.is_empty() => {
                    Ok(TensorData::new(tensor.int32_data, shape))
                }
                DType::I32 if expected_elems == 0 => {
                    Ok(TensorData::new(Vec::<i32>::new(), shape))
                }
                DType::I64 if !tensor.int64_data.is_empty() => {
                    Ok(TensorData::new(tensor.int64_data, shape))
                }
                DType::I64 if expected_elems == 0 => {
                    Ok(TensorData::new(Vec::<i64>::new(), shape))
                }
                DType::Bool if !tensor.int32_data.is_empty() => {
                    let data: Vec<bool> = tensor.int32_data.into_iter().map(|x| x != 0).collect();
                    Ok(TensorData::new(data, shape))
                }
                DType::Bool if expected_elems == 0 => {
                    Ok(TensorData::new(Vec::<bool>::new(), shape))
                }
                DType::U8 => {
                    // accept weird exporters that stuff zp as int32_data
                    if !tensor.int32_data.is_empty() {
                        let data: Vec<u8> =
                            tensor.int32_data.into_iter().map(|x| x as u8).collect();
                        Ok(TensorData::new(data, shape))
                    } else if expected_elems == 0 {
                        Ok(TensorData::new(Vec::<u8>::new(), shape))
                    } else {
                        Err(ParseError::VariantNotFound("no data for UINT8".into()))
                    }
                }
                DType::I8 => {
                    if !tensor.int32_data.is_empty() {
                        let data: Vec<i8> =
                            tensor.int32_data.into_iter().map(|x| x as i8).collect();
                        Ok(TensorData::new(data, shape))
                    } else if expected_elems == 0 {
                        Ok(TensorData::new(Vec::<i8>::new(), shape))
                    } else {
                        Err(ParseError::VariantNotFound("no data for INT8".into()))
                    }
                }
                _ => Err(ParseError::VariantNotFound(format!(
                    "empty/unsupported payload for {:?}",
                    elem
                ))),
            }
        }
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

pub fn convert_node_proto(node: &NodeProto, graph_data: &GraphState) -> Node {
    let name = node.name.clone();

    log::debug!("Converting ONNX node with type {:?}", node.op_type.as_str());

    let inputs = node.input.iter().map(|x| graph_data.init_in(x)).collect();

    let outputs = node
        .output
        .iter()
        .map(|output_name| {
            let mut arg = Argument::new(output_name.to_string());
            // If this output is a graph output, use its type from the graph
            if let Some(graph_output_type) = graph_data.get_output_type(output_name) {
                arg.ty = graph_output_type.clone();
            }
            arg
        })
        .collect();

    let attrs = convert_vec_attrs_proto(node.attribute.clone());

    let node_type = NodeType::from_str(node.op_type.as_str()).expect("Unknown node type");

    Node {
        node_type,
        name,
        inputs,
        outputs,
        attrs,
        config: None,
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
                dtype: elem_type,
                static_shape,
            })
        };

        Ok(Argument {
            ty,
            name,
            data_id: None,
            value_source: crate::ir::ValueSource::Dynamic, // Graph inputs/outputs are runtime values
            value_store: None,
        })
    }
}
