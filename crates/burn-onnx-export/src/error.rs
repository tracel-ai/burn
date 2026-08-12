use burn_ir::TensorId;
use thiserror::Error;

/// Errors produced while capturing, resolving, or lowering an export graph.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ExportError {
    /// Two traces do not describe the same computation.
    #[error("dynamic graph differs at operation {operation}: {reason}")]
    DynamicGraphMismatch {
        /// Operation index, or the operation count when lengths differ.
        operation: usize,
        /// Human-readable mismatch detail.
        reason: String,
    },
    /// A runtime shape expression could not be recovered conservatively.
    #[error("dynamic shape was lost for tensor {tensor} at axis {axis}: {reason}")]
    DynamicShapeLost {
        /// Tensor whose shape is unresolved.
        tensor: TensorId,
        /// Unresolved axis.
        axis: usize,
        /// Human-readable ambiguity detail.
        reason: String,
    },
    /// An operation has no ONNX lowering.
    #[error("unsupported operation at index {operation}: {kind}")]
    UnsupportedOperation {
        /// Operation index.
        operation: usize,
        /// Burn operation kind.
        kind: String,
    },
    /// A referenced tensor value is unavailable.
    #[error("missing value for tensor {0}")]
    MissingValue(TensorId),
    /// An initialized value disagrees with the tensor metadata in the graph.
    #[error("invalid value for tensor {tensor}: {reason}")]
    InvalidValue {
        /// Tensor identifier.
        tensor: TensorId,
        /// Shape or dtype mismatch detail.
        reason: String,
    },
    /// Graph inputs or outputs are inconsistent.
    #[error("invalid graph boundary: {0}")]
    InvalidBoundary(String),
    /// A tensor dtype has no supported ONNX representation.
    #[error("unsupported dtype for tensor {tensor}: {dtype}")]
    UnsupportedDType {
        /// Tensor identifier.
        tensor: TensorId,
        /// Debug representation of the dtype.
        dtype: String,
    },
    /// Encoding or writing the ONNX model failed.
    #[error("serialization failed: {0}")]
    Serialization(String),
    /// Writing a serialized ONNX model to disk failed.
    #[error("failed to write ONNX model to `{path}`: {reason}")]
    FileWrite {
        /// Destination path.
        path: String,
        /// Underlying I/O error.
        reason: String,
    },
}
