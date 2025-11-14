//! ONNX attribute values
//!
//! This module contains the AttributeValue enum which represents various types
//! of attributes that can be attached to ONNX nodes.

use std::collections::HashMap;

use burn_tensor::TensorData;

use crate::ir::{OnnxGraph, OnnxGraphBuilder};

/// The type of an attribute.
#[derive(Debug, Clone)]
pub(crate) enum AttributeValue {
    Float32(f32),
    Float32s(Vec<f32>),
    Int64(i64),
    Int64s(Vec<i64>),
    String(String),
    #[allow(dead_code)]
    Strings(Vec<String>),
    Tensor(TensorData),
    #[allow(dead_code)]
    Tensors(Vec<TensorData>),
    /// Graph attribute - holds OnnxGraphBuilder during processing, converts to OnnxGraph later
    GraphBuilder(OnnxGraphBuilder),
    /// Multiple graph attributes
    GraphBuilders(Vec<OnnxGraphBuilder>),
    /// Final graph after conversion (used in final Node enum)
    Graph(OnnxGraph),
    /// Final graphs after conversion (used in final Node enum)
    #[allow(dead_code)]
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

    #[allow(dead_code)]
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

    #[allow(dead_code)]
    pub fn into_strings(self) -> Vec<String> {
        if let AttributeValue::Strings(elem) = self {
            elem
        } else {
            panic!("Expected Strings, got {self:?}");
        }
    }

    #[allow(dead_code)]
    pub fn into_tensors(self) -> Vec<TensorData> {
        if let AttributeValue::Tensors(elem) = self {
            elem
        } else {
            panic!("Expected Tensors, got {self:?}");
        }
    }

    #[allow(dead_code)]
    pub fn into_graph(self) -> OnnxGraph {
        if let AttributeValue::Graph(elem) = self {
            elem
        } else {
            panic!("Expected Graph, got {self:?}");
        }
    }

    #[allow(dead_code)]
    pub fn into_graphs(self) -> Vec<OnnxGraph> {
        if let AttributeValue::Graphs(elem) = self {
            elem
        } else {
            panic!("Expected Graphs, got {self:?}");
        }
    }
}
