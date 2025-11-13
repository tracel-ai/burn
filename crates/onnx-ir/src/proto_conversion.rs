use std::str::{FromStr, from_utf8};

use super::graph_state::GraphState;
use super::ir::{
    ArgType, Argument, AttributeValue, Attributes, NodeBuilder, NodeType, TensorData,
    TensorDataExt, TensorType,
};
use super::protos::{
    AttributeProto, NodeProto, TensorProto, TensorShapeProto, ValueInfoProto,
    attribute_proto::AttributeType, tensor_proto::DataType as DT,
    tensor_shape_proto::dimension::Value,
};

use burn_tensor::DType;
use protobuf::Enum;

/// Minimum required ONNX opset version
pub const MIN_OPSET_VERSION: usize = 16;

/// Error type for parsing ONNX model
#[derive(Debug)]
pub enum ParseError {
    VariantNotFound(String),
}

/// Sanitize ONNX names to be valid Rust identifiers in snake_case
///
/// This function converts ONNX variable names (which can contain special characters
/// like ':',  '/', '.', etc.) into valid Rust identifiers by:
/// 1. Keeping empty strings as-is (they represent optional inputs in ONNX)
/// 2. Converting to snake_case (lowercase with underscores)
/// 3. Replacing invalid characters with underscores
/// 4. Prepending an underscore if the name starts with a digit
///
/// Examples:
/// - "" -> "" (empty strings represent optional inputs)
/// - "input:0" -> "input_0"
/// - "jax2tf/model/layer.weight" -> "jax2tf_model_layer_weight"
/// - "123tensor" -> "_123tensor"
/// - "onnx__GlobalAveragePool_0" -> "onnx_global_average_pool_0"
/// - "MyVariable" -> "my_variable"
pub fn sanitize_name(name: &str) -> String {
    // Empty strings represent optional inputs in ONNX - keep them as-is
    if name.is_empty() {
        return String::new();
    }

    let mut result = String::with_capacity(name.len() * 2);
    let mut prev_is_lower = false;
    let mut prev_is_underscore = false;

    for (i, c) in name.chars().enumerate() {
        if c == '_' {
            // Keep existing underscores, but avoid consecutive ones
            if !prev_is_underscore || i == 0 {
                result.push('_');
                prev_is_underscore = true;
            }
            prev_is_lower = false;
        } else if c.is_ascii_alphanumeric() {
            // Insert underscore before uppercase letters that follow lowercase letters
            if c.is_ascii_uppercase() && prev_is_lower && !prev_is_underscore {
                result.push('_');
            }

            result.push(c.to_ascii_lowercase());
            prev_is_lower = c.is_ascii_lowercase();
            prev_is_underscore = false;
        } else {
            // Replace invalid characters with underscores, but avoid consecutive underscores
            if !prev_is_underscore && i > 0 {
                result.push('_');
                prev_is_underscore = true;
            }
            prev_is_lower = false;
        }
    }

    // Remove trailing underscores (but not if the entire string is underscores)
    while result.ends_with('_') && result.len() > 1 {
        // Check if removing this underscore would leave us with something
        let check = result.trim_end_matches('_');
        if check.is_empty() {
            // All underscores, keep them
            break;
        }
        result.pop();
    }

    // Ensure the first character is valid to start an identifier
    if !result.is_empty() && !result.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_') {
        result = format!("_{result}");
    }

    result
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
            match elem {
                DType::F32 => Ok(TensorData::new(tensor.float_data, shape)),
                DType::F64 => Ok(TensorData::new(tensor.double_data, shape)),
                DType::I32 => Ok(TensorData::new(tensor.int32_data, shape)),
                DType::I64 => Ok(TensorData::new(tensor.int64_data, shape)),
                DType::Bool => {
                    let data: Vec<bool> = tensor.int32_data.into_iter().map(|x| x != 0).collect();
                    Ok(TensorData::new(data, shape))
                }
                DType::U8 => {
                    // accept weird exporters that stuff zp as int32_data
                    let data: Vec<u8> = tensor.int32_data.into_iter().map(|x| x as u8).collect();
                    Ok(TensorData::new(data, shape))
                }
                DType::I8 => {
                    let data: Vec<i8> = tensor.int32_data.into_iter().map(|x| x as i8).collect();
                    Ok(TensorData::new(data, shape))
                }
                DType::F16 => Ok(TensorData::new(Vec::<half::f16>::new(), shape)),
                DType::U16 => Ok(TensorData::new(Vec::<u16>::new(), shape)),
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

            // Graph attributes (used by If, Loop, Scan)
            AttributeType::GRAPH => {
                // Note: We can't convert the graph here without the opset version
                // This conversion will be handled during node processing where we have access to opset
                // For now, we'll store a placeholder and do the actual conversion in the If processor
                panic!(
                    "Graph attributes should be converted during node processing, not during proto conversion"
                )
            }
            AttributeType::FLOATS => AttributeValue::Float32s(attr.floats),
            AttributeType::INTS => AttributeValue::Int64s(attr.ints),
            AttributeType::STRINGS => AttributeValue::Strings(to_string_vec(attr.strings)),
            AttributeType::TENSORS => {
                AttributeValue::Tensors(convert_vec_tensor_proto(attr.tensors)?)
            }
            AttributeType::GRAPHS => {
                panic!(
                    "Graphs attributes should be converted during node processing, not during proto conversion"
                )
            }
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
/// Skips GRAPH and GRAPHS attributes as they need special handling with opset version
pub fn convert_vec_attrs_proto(attrs: Vec<AttributeProto>) -> Attributes {
    let mut result = Attributes::new();
    for attr in attrs {
        // Skip GRAPH/GRAPHS attributes - they'll be handled separately with opset version
        if let Ok(attr_type) = attr.type_.enum_value()
            && (attr_type == AttributeType::GRAPH || attr_type == AttributeType::GRAPHS)
        {
            continue;
        }
        result.insert(attr.name.clone(), AttributeValue::try_from(attr).unwrap());
    }
    result
}

pub fn convert_node_proto(node: &NodeProto, graph_data: &GraphState) -> NodeBuilder {
    let name = sanitize_name(&node.name);

    let inputs = node.input.iter().map(|x| graph_data.init_in(x)).collect();

    let outputs = node
        .output
        .iter()
        .map(|output_name| {
            // Sanitize the output name for Rust compatibility
            let mut arg = Argument::new(sanitize_name(output_name));
            // Try to get type from: 1) graph outputs, 2) value_info (intermediate values)
            // Note: lookups use original ONNX names (unsanitized)
            if let Some(graph_output_type) = graph_data.get_output_type(output_name) {
                arg.ty = graph_output_type.clone();
            } else if let Some(value_info_type) = graph_data.get_value_info_type(output_name) {
                arg.ty = value_info_type.clone();
            }
            arg
        })
        .collect();

    let attrs = convert_vec_attrs_proto(node.attribute.clone());

    let node_type = NodeType::from_str(node.op_type.as_str()).expect("Unknown node type");

    NodeBuilder {
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

/// Convert graph attributes from NodeProto for control flow nodes (If, Loop, Scan)
/// This must be called with opset_version during node processing
///
/// If parent_registry is provided, it will be used to ensure unique names across nested subgraphs.
/// Otherwise, a new registry is created for sibling subgraphs.
pub fn convert_graph_attributes(
    node_proto: &NodeProto,
    opset_version: usize,
    parent_registry: Option<crate::graph_state::NameRegistry>,
) -> Attributes {
    use crate::pipeline::build_graph_from_proto_with_registry;

    let mut result = Attributes::new();

    // Use parent registry if provided, otherwise create a new one for sibling subgraphs
    // This ensures node names are unique across nested levels and sibling branches
    let name_registry = parent_registry.unwrap_or_default();

    for attr in &node_proto.attribute {
        if let Ok(attr_type) = attr.type_.enum_value() {
            match attr_type {
                AttributeType::GRAPH => {
                    if let Some(graph_proto) = attr.g.as_ref() {
                        let onnx_graph = build_graph_from_proto_with_registry(
                            graph_proto,
                            opset_version,
                            Some(name_registry.clone()),
                        )
                        .expect("Failed to build subgraph from ONNX GraphProto");
                        result.insert(attr.name.clone(), AttributeValue::Graph(onnx_graph));
                    }
                }
                AttributeType::GRAPHS => {
                    let graphs: Vec<_> = attr
                        .graphs
                        .iter()
                        .map(|graph_proto| {
                            build_graph_from_proto_with_registry(
                                graph_proto,
                                opset_version,
                                Some(name_registry.clone()),
                            )
                            .expect("Failed to build subgraph from ONNX GraphProto")
                        })
                        .collect();
                    result.insert(attr.name.clone(), AttributeValue::Graphs(graphs));
                }
                _ => {}
            }
        }
    }
    result
}

impl TryFrom<ValueInfoProto> for Argument {
    type Error = ParseError;

    fn try_from(value: ValueInfoProto) -> Result<Argument, Self::Error> {
        let name = sanitize_name(&value.name);
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
            value_source: crate::ir::ValueSource::Dynamic, // Graph inputs/outputs are runtime values
            value_store: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_name_basic() {
        // Already snake_case
        assert_eq!(sanitize_name("valid_name"), "valid_name");
        assert_eq!(sanitize_name("_underscore"), "_underscore");
        assert_eq!(sanitize_name("a"), "a");

        // Convert to snake_case
        assert_eq!(sanitize_name("ValidName123"), "valid_name123");
        assert_eq!(sanitize_name("MyVariable"), "my_variable");
        assert_eq!(sanitize_name("HTTPResponse"), "httpresponse");
    }

    #[test]
    fn test_sanitize_name_special_chars() {
        // TensorFlow/JAX style names with colons and slashes
        assert_eq!(sanitize_name("input:0"), "input_0");
        assert_eq!(sanitize_name("layer/weight"), "layer_weight");
        assert_eq!(sanitize_name("jax2tf/model:0"), "jax2tf_model_0");

        // ONNX names with dots and dashes
        assert_eq!(sanitize_name("bert.encoder.layer"), "bert_encoder_layer");
        assert_eq!(sanitize_name("layer-norm"), "layer_norm");

        // Complex real-world example from GitHub issue #2878
        assert_eq!(
            sanitize_name("jax2tf_rhs_/pjit_silu_/Const_2:0"),
            "jax2tf_rhs_pjit_silu_const_2_0"
        );
    }

    #[test]
    fn test_sanitize_name_camel_to_snake() {
        // Convert CamelCase and PascalCase to snake_case
        assert_eq!(
            sanitize_name("onnx__GlobalAveragePool_0"),
            "onnx_global_average_pool_0"
        );
        assert_eq!(sanitize_name("onnx__Gemm_0"), "onnx_gemm_0");
        assert_eq!(sanitize_name("onnx__Greater_0"), "onnx_greater_0");
        assert_eq!(sanitize_name("MyClassName"), "my_class_name");
        assert_eq!(sanitize_name("HTTPSConnection"), "httpsconnection");
    }

    #[test]
    fn test_sanitize_name_starts_with_digit() {
        assert_eq!(sanitize_name("123tensor"), "_123tensor");
        assert_eq!(sanitize_name("0input"), "_0input");
    }

    #[test]
    fn test_sanitize_name_unicode() {
        // Unicode characters should be replaced with underscores
        assert_eq!(sanitize_name("tensor™"), "tensor");
        assert_eq!(sanitize_name("input€output"), "input_output");
    }

    #[test]
    fn test_sanitize_name_empty_and_edge_cases() {
        // Empty strings represent optional inputs in ONNX - should remain empty
        assert_eq!(sanitize_name(""), "");

        assert_eq!(sanitize_name("_"), "_");

        // Consecutive underscores are collapsed to single underscore
        assert_eq!(sanitize_name("___"), "_");
        assert_eq!(sanitize_name("a__b"), "a_b");

        // All special chars become single underscore
        assert_eq!(sanitize_name(":/:"), "_");

        // Consecutive special chars become single underscore
        assert_eq!(sanitize_name("a:::b"), "a_b");

        // Trailing underscores removed
        assert_eq!(sanitize_name("name_:"), "name");
    }
}
