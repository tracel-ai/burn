//! Export captured Burn operation graphs to ONNX.
//!
//! Shape validation and resolution deliberately precede ONNX lowering. This
//! keeps trace-based inference replaceable by a future symbolic capture pass.

mod error;
mod lower;
mod shape;
mod validate;

pub use error::ExportError;
pub use lower::{ONNX_IR_VERSION, ONNX_OPSET_VERSION, export_graph};
pub use shape::{
    AxisSpec, InputSpec, PairedTraceShapeResolver, ResolvedExportGraph, ResolvedShape, ShapeExpr,
    ShapeResolver, StaticShapeResolver,
};
pub use validate::GraphStructureValidator;
