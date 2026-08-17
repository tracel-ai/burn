//! Exporter intermediate representation produced by shape resolution.
//!
//! This representation is the contract between graph capture/shape analysis
//! and ONNX lowering. Lowering consumes explicit shape expressions and does not
//! need to know whether they came from static, paired-trace, or future symbolic
//! resolution.

use burn_ir::{GraphIr, TensorId};

/// Symbolic axis attached to a captured runtime input or graph output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynamicAxis {
    /// Tensor carrying the symbolic dimension.
    pub tensor: TensorId,
    /// Axis within the tensor.
    pub axis: usize,
    /// ONNX symbolic dimension name.
    pub symbol: String,
}

/// An explicit ONNX-compatible dimension expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeExpr {
    /// Constant dimension.
    Static(usize),
    /// Dimension of a declared runtime input.
    InputDim { input: TensorId, axis: usize },
    /// Dimension of an intermediate or source tensor.
    TensorDim { tensor: TensorId, axis: usize },
    /// Element-count-preserving inferred dimension (`-1` in ONNX reshape).
    Infer,
}

/// Resolved shape operand for one shape-sensitive operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedShape {
    /// Operation index in [`GraphIr::operations`].
    pub operation: usize,
    /// Output tensor receiving the shape.
    pub tensor: TensorId,
    /// Dimension expressions in axis order.
    pub dimensions: Vec<ShapeExpr>,
}

/// Captured graph plus the explicit shape information required by lowering.
///
/// The graph has already passed the structural checks appropriate to its shape
/// resolver. `shapes` contains runtime expressions for shape operands such as
/// reshape targets, while `dynamic_axes` controls symbolic ONNX boundary
/// dimensions.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedExportGraph {
    /// Validated captured graph.
    pub graph: GraphIr,
    /// Resolved shape-sensitive operands.
    pub shapes: Vec<ResolvedShape>,
    /// Symbolic dimensions declared on graph boundaries.
    pub dynamic_axes: Vec<DynamicAxis>,
}
