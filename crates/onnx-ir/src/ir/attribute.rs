//! ONNX attribute values
//!
//! This module contains the AttributeValue enum which represents various types
//! of attributes that can be attached to ONNX nodes.

use std::collections::HashMap;

use super::argument::{ArgType, Argument, TensorType, ValueSource};
use super::graph::OnnxGraph;
use super::tensor_data_ext::TensorDataExt;
use burn_tensor::{DType, TensorData};

/// The type of an attribute.
#[derive(Debug, Clone)]
pub enum AttributeValue {
    Float32(f32),
    Float32s(Vec<f32>),
    Int64(i64),
    Int64s(Vec<i64>),
    String(String),
    Strings(Vec<String>),
    Tensor(TensorData),
    Tensors(Vec<TensorData>),
    Graph(OnnxGraph),
    Graphs(Vec<OnnxGraph>),
}

pub type Attributes = HashMap<String, AttributeValue>;

impl AttributeValue {
    pub fn into_f32(self) -> f32 {
        if let AttributeValue::Float32(elem) = self {
            elem
        } else {
            panic!("Expected Float32, got {self:?}");
        }
    }

    pub fn into_i32(self) -> i32 {
        if let AttributeValue::Int64(elem) = self {
            elem as i32
        } else {
            panic!("Expected Int32, got {self:?}");
        }
    }

    pub fn into_i64(self) -> i64 {
        if let AttributeValue::Int64(elem) = self {
            elem
        } else {
            panic!("Expected Int64, got {self:?}");
        }
    }

    pub fn into_string(self) -> String {
        if let AttributeValue::String(elem) = self {
            elem
        } else {
            panic!("Expected String, got {self:?}");
        }
    }

    pub fn into_tensor(self) -> TensorData {
        if let AttributeValue::Tensor(elem) = self {
            elem
        } else {
            panic!("Expected Tensor, got {self:?}");
        }
    }

    pub fn into_f32s(self) -> Vec<f32> {
        if let AttributeValue::Float32s(elem) = self {
            elem
        } else {
            panic!("Expected Float32s, got {self:?}");
        }
    }

    pub fn into_i64s(self) -> Vec<i64> {
        if let AttributeValue::Int64s(elem) = self {
            elem
        } else {
            panic!("Expected Int64s, got {self:?}");
        }
    }

    pub fn into_strings(self) -> Vec<String> {
        if let AttributeValue::Strings(elem) = self {
            elem
        } else {
            panic!("Expected Strings, got {self:?}");
        }
    }

    pub fn into_tensors(self) -> Vec<TensorData> {
        if let AttributeValue::Tensors(elem) = self {
            elem
        } else {
            panic!("Expected Tensors, got {self:?}");
        }
    }

    pub fn into_graph(self) -> OnnxGraph {
        if let AttributeValue::Graph(elem) = self {
            elem
        } else {
            panic!("Expected Graph, got {self:?}");
        }
    }

    pub fn into_graphs(self) -> Vec<OnnxGraph> {
        if let AttributeValue::Graphs(elem) = self {
            elem
        } else {
            panic!("Expected Graphs, got {self:?}");
        }
    }
}

/// Convert AttributeValue to an Argument
impl From<AttributeValue> for Argument {
    fn from(attr: AttributeValue) -> Argument {
        let name = "".to_string();

        match attr {
            AttributeValue::Float32(_value) => Argument {
                ty: ArgType::Scalar(DType::F32),
                name,
                data_id: None,
                value_source: ValueSource::Optional,
                value_store: None,
            },
            AttributeValue::Float32s(values) => Argument {
                ty: ArgType::Tensor(TensorType {
                    rank: 1,
                    dtype: DType::F32,
                    static_shape: Some(vec![values.len()]),
                }),
                name,
                data_id: None,
                value_source: ValueSource::Optional,
                value_store: None,
            },
            AttributeValue::Int64(_value) => Argument {
                ty: ArgType::Scalar(DType::I64),
                name,
                data_id: None,
                value_source: ValueSource::Optional,
                value_store: None,
            },
            AttributeValue::Int64s(values) => Argument {
                ty: ArgType::Tensor(TensorType {
                    rank: 1,
                    dtype: DType::I64,
                    static_shape: Some(vec![values.len()]),
                }),
                name,
                data_id: None,
                value_source: ValueSource::Optional,
                value_store: None,
            },
            AttributeValue::String(_value) => {
                panic!(
                    "String type not supported in DType - use AttributeValue directly for strings"
                )
            }
            AttributeValue::Strings(_values) => {
                panic!(
                    "String type not supported in DType - use AttributeValue directly for strings"
                )
            }
            AttributeValue::Tensor(tensor) => {
                if tensor.shape.is_empty() {
                    // Handle scalar tensors by converting them to scalar arguments
                    Argument {
                        ty: ArgType::Scalar(tensor.elem_type()),
                        name,
                        data_id: None,
                        value_source: ValueSource::Optional,
                        value_store: None,
                    }
                } else {
                    // Convert tensor to argument
                    Argument {
                        ty: ArgType::Tensor(TensorType {
                            rank: tensor.shape.len(),
                            dtype: tensor.elem_type(),
                            static_shape: Some(tensor.shape.to_vec()),
                        }),
                        name,
                        data_id: None,
                        value_source: ValueSource::Optional,
                        value_store: None,
                    }
                }
            }
            AttributeValue::Graph(_) => {
                panic!(
                    "Graph attributes cannot be converted to Argument - they should be accessed directly"
                )
            }
            AttributeValue::Graphs(_) => {
                panic!(
                    "Graphs attributes cannot be converted to Argument - they should be accessed directly"
                )
            }
            _ => panic!("Unsupported attribute type"),
        }
    }
}
