//! Export captured Burn operation graphs to ONNX.
//!
//! Shape validation and resolution deliberately precede ONNX lowering. This
//! keeps trace-based inference replaceable by a future symbolic capture pass.

extern crate alloc;

mod error;
mod exporter;
mod lower;
mod shape;
mod validate;

pub use error::ExportError;
pub use exporter::{ExportInput, ExportValues, OnnxExporter};
pub use lower::{
    MAX_EMBEDDED_PROTOBUF_BYTES, ONNX_IR_VERSION, ONNX_OPSET_VERSION, export_graph,
    export_graph_with_bindings, export_graph_with_values,
};
pub use shape::{
    AxisSpec, DynamicAxis, InputSpec, PairedTraceShapeResolver, ResolvedExportGraph, ResolvedShape,
    ShapeExpr, ShapeResolver, StaticShapeResolver,
};
pub use validate::GraphStructureValidator;
