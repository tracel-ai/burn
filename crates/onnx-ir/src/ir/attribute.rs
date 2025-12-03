//! ONNX attribute values
//!
//! This module contains the AttributeValue enum which represents various types
//! of attributes that can be attached to ONNX nodes.

use std::collections::HashMap;
use std::sync::Arc;

use burn_tensor::TensorData;

use crate::ir::{OnnxGraph, OnnxGraphBuilder};
use crate::protos::GraphProto;

/// Deferred subgraph that is built lazily during type inference.
///
/// ## Why Deferred?
///
/// ONNX control flow nodes (If, Loop, Scan) contain subgraphs that can reference
/// values from the parent graph's scope. For example:
///
/// ```text
/// x = Conv(input, weights)      // x has type Tensor[F32, rank=4]
/// y = If(condition) {
///     then_branch: Add(x, bias)  // References 'x' from parent scope
///     else_branch: Mul(x, scale)
/// }
/// ```
///
/// The subgraph's `Add(x, bias)` needs to know the type of `x` for type inference,
/// but `x`'s type is only determined after processing the parent's `Conv` node.
///
/// ## Solution: Lazy Building
///
/// 1. **Parse phase**: Store the raw `GraphProto` in `DeferredGraph`
/// 2. **Type inference phase**: When processing If/Loop/Scan, outer-scope types are known
/// 3. **Build phase**: Call `build_graph_with_outer_scope(types)` with resolved types
///
/// This lazy evaluation pattern ensures subgraphs have access to all type information
/// from the parent scope when they are finally built.
#[derive(Debug, Clone)]
pub struct DeferredGraph {
    /// The raw ONNX GraphProto (wrapped in Arc for cheap cloning)
    pub proto: Arc<GraphProto>,
    /// The opset version to use when building the subgraph
    pub opset_version: usize,
    /// Name registry for unique node naming across subgraphs
    pub name_registry: Option<crate::graph_state::NameRegistry>,
}

/// A map of outer-scope value names to their resolved types
pub type OuterScopeTypes = std::collections::HashMap<String, crate::ir::ArgType>;

impl DeferredGraph {
    /// Build the subgraph from the deferred GraphProto with access to outer scope types.
    ///
    /// This should be called during type inference when all outer-scope
    /// references have been resolved. The `outer_scope` map provides types
    /// for values that the subgraph references from the parent graph.
    pub fn build_with_outer_scope(
        &self,
        outer_scope: OuterScopeTypes,
    ) -> Result<OnnxGraphBuilder, crate::pipeline::Error> {
        crate::pipeline::build_graph_builder_from_proto_with_outer_scope(
            &self.proto,
            self.opset_version,
            self.name_registry.clone(),
            outer_scope,
        )
    }

    /// Build and finalize the subgraph into an OnnxGraph with outer scope types.
    pub fn build_graph_with_outer_scope(
        &self,
        outer_scope: OuterScopeTypes,
    ) -> Result<OnnxGraph, crate::pipeline::Error> {
        let builder = self.build_with_outer_scope(outer_scope)?;
        Ok(builder.convert_to_graph(self.opset_version))
    }

    /// Build the subgraph from the deferred GraphProto without outer scope types.
    ///
    /// Useful for simple subgraphs that don't reference outer-scope values.
    #[allow(dead_code)]
    pub fn build(&self) -> Result<OnnxGraphBuilder, crate::pipeline::Error> {
        self.build_with_outer_scope(OuterScopeTypes::new())
    }

    /// Build and finalize the subgraph into an OnnxGraph without outer scope types.
    #[allow(dead_code)]
    pub fn build_graph(&self) -> Result<OnnxGraph, crate::pipeline::Error> {
        let builder = self.build()?;
        Ok(builder.convert_to_graph(self.opset_version))
    }
}

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
    /// Deferred graph attribute - raw GraphProto to be built during type inference
    DeferredGraph(DeferredGraph),
    /// Multiple deferred graphs (for ONNX GRAPHS attributes)
    #[allow(dead_code)]
    DeferredGraphs(Vec<DeferredGraph>),
    /// Final graph after conversion (used in final Node enum)
    /// Note: Constructed via DeferredGraph::build_graph_with_outer_scope(), not directly
    #[allow(dead_code)]
    Graph(OnnxGraph),
    /// Multiple final graphs (for ONNX GRAPHS attributes)
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
