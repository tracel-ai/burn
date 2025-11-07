//! ONNX to IR conversion pipeline orchestrator
//!
//! This module provides the high-level orchestration of the ONNX conversion process.
//! It clearly shows the entire conversion flow from start to finish.

use std::{fmt, fs::File, path::Path};

use protobuf::Message;

use crate::{
    ir::OnnxGraph, processor::ProcessError, proto_conversion::MIN_OPSET_VERSION,
    protos::ModelProto, util::verify_opsets,
};

use super::phases::{
    finalization, initialization, node_conversion, post_processing, type_inference,
};

/// Errors that can occur when parsing ONNX models
#[derive(Debug)]
pub enum OnnxIrError {
    /// Failed to open or read the ONNX file
    Io { path: String, error: std::io::Error },

    /// Failed to parse ONNX protobuf format
    InvalidFormat { path: Option<String>, error: String },

    /// ONNX opset version is not supported
    UnsupportedOpset { found: i64, minimum_required: i64 },

    /// Model graph nodes are not topologically sorted (ONNX spec violation)
    InvalidGraphStructure { reason: String },

    /// Missing required opset version for default domain
    MissingOpsetVersion,

    /// Type inference failed during IR conversion
    TypeInference(ProcessError),

    /// Generic processing error
    Processing(ProcessError),
}

impl fmt::Display for OnnxIrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OnnxIrError::Io { path, error } => {
                write!(f, "Failed to open ONNX file '{}': {}", path, error)
            }
            OnnxIrError::InvalidFormat { path, error } => {
                if let Some(p) = path {
                    write!(f, "Invalid ONNX format in '{}': {}", p, error)
                } else {
                    write!(f, "Invalid ONNX format: {}", error)
                }
            }
            OnnxIrError::UnsupportedOpset {
                found,
                minimum_required,
            } => {
                write!(
                    f,
                    "Unsupported ONNX opset version {}. Requires opset {} or higher. \
                    See documentation for upgrade instructions.",
                    found, minimum_required
                )
            }
            OnnxIrError::InvalidGraphStructure { reason } => {
                write!(f, "Invalid ONNX graph structure: {}", reason)
            }
            OnnxIrError::MissingOpsetVersion => {
                write!(
                    f,
                    "ONNX model must specify opset version for default domain"
                )
            }
            OnnxIrError::TypeInference(e) => {
                write!(f, "Type inference failed: {:?}", e)
            }
            OnnxIrError::Processing(e) => {
                write!(f, "Processing error: {:?}", e)
            }
        }
    }
}

impl std::error::Error for OnnxIrError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            OnnxIrError::Io { error, .. } => Some(error),
            _ => None,
        }
    }
}

impl From<ProcessError> for OnnxIrError {
    fn from(error: ProcessError) -> Self {
        OnnxIrError::Processing(error)
    }
}

/// Parse an ONNX file and convert to IR
///
/// # Errors
///
/// Returns an error if:
/// - File cannot be opened or read
/// - File is not valid ONNX protobuf format
/// - ONNX opset version is less than 16
/// - Graph nodes are not topologically sorted
/// - Type inference fails
pub fn parse_onnx(onnx_path: &Path) -> Result<OnnxGraph, OnnxIrError> {
    log::info!("Parsing ONNX file: {}", onnx_path.display());

    // Load and validate model
    let mut file = File::open(onnx_path).map_err(|error| OnnxIrError::Io {
        path: onnx_path.display().to_string(),
        error,
    })?;

    let model: ModelProto =
        Message::parse_from_reader(&mut file).map_err(|e| OnnxIrError::InvalidFormat {
            path: Some(onnx_path.display().to_string()),
            error: e.to_string(),
        })?;

    if !verify_opsets(&model.opset_import, MIN_OPSET_VERSION) {
        // Find the actual opset version for better error message
        let found_version = model
            .opset_import
            .iter()
            .find(|opset| opset.domain.is_empty())
            .map(|opset| opset.version)
            .unwrap_or(0);

        return Err(OnnxIrError::UnsupportedOpset {
            found: found_version,
            minimum_required: MIN_OPSET_VERSION,
        });
    }

    // ONNX nodes must be topologically sorted per spec:
    // https://github.com/onnx/onnx/blob/main/docs/IR.md#graphs
    // This is a runtime check (not debug_assert) to catch malformed models in production
    if !model.graph.node.is_top_sorted() {
        return Err(OnnxIrError::InvalidGraphStructure {
            reason: "Nodes are not topologically sorted (ONNX spec violation)".to_string(),
        });
    }

    let graph = build_graph(&model)?;

    log::info!("Finished parsing ONNX file: {}", onnx_path.display());
    Ok(graph)
}

/// Build IR graph from ONNX model through 5 phases:
/// 1. Initialization 2. Node Conversion 3. Type Inference 4. Post-processing 5. Finalization
///
/// # Errors
///
/// Returns an error if:
/// - Missing opset version for default domain
/// - Type inference fails
pub fn build_graph(model: &ModelProto) -> Result<OnnxGraph, OnnxIrError> {
    log::debug!(" PHASE 1: Initialization ");
    let state_rc = initialization::initialize(model);

    log::debug!(" PHASE 2: Node Conversion ");
    node_conversion::convert_nodes(model, &state_rc);

    log::debug!(" PHASE 3: Type Inference ");
    let opset_version = extract_opset_version(model)?;
    type_inference::infer_types(&state_rc, opset_version).map_err(OnnxIrError::TypeInference)?;

    log::debug!(" PHASE 4: Post-processing ");
    let (mut nodes, inputs, mut outputs) = post_processing::post_process(&state_rc);

    log::debug!(" PHASE 5: Finalization ");
    Ok(finalization::finalize(
        &mut nodes,
        inputs,
        &mut outputs,
        state_rc,
    ))
}

/// Extract opset version from model (default ONNX domain)
fn extract_opset_version(model: &ModelProto) -> Result<usize, OnnxIrError> {
    model
        .opset_import
        .iter()
        .find(|opset| opset.domain.is_empty())
        .map(|opset| opset.version as usize)
        .ok_or(OnnxIrError::MissingOpsetVersion)
}

/// Trait for checking if a list of nodes is topologically sorted
pub(crate) trait TopologicalSortable {
    fn is_top_sorted(&self) -> bool;
}

use crate::protos::NodeProto;

impl TopologicalSortable for Vec<NodeProto> {
    fn is_top_sorted(&self) -> bool {
        // Iterate over each node in the vector
        for (node_position, node) in self.iter().enumerate() {
            // Iterate over each output of the node
            for output in &node.output {
                // If the output is empty, we don't want to check the rest of the graph, inputs and outputs that are optional
                // can end up as empty strings, so we can't use that as a reason to count the graph as not sorted
                if output.is_empty() {
                    continue;
                }
                // Iterate over each other node in the vector
                for (other_node_position, other_node) in self.iter().enumerate() {
                    // If the other node has an input that matches the current output
                    if other_node.input.contains(output) {
                        // If the position of the current node is greater than the position of the other node
                        if node_position > other_node_position {
                            // The vector is not topologically sorted
                            return false;
                        }
                    }
                }
            }
        }

        // The vector is topologically sorted
        true
    }
}
