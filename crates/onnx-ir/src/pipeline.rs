//! ONNX to IR conversion pipeline orchestrator
//!
//! This module provides the high-level orchestration of the ONNX conversion process.
//! It clearly shows the entire conversion flow from start to finish.
//!
//! # Zero-Copy Loading
//!
//! When the `mmap` feature is enabled (default), files are memory-mapped for zero-copy
//! tensor loading. This significantly reduces memory usage for large models.
//!
//! # Usage
//!
//! ```ignore
//! use onnx_ir::OnnxGraphBuilder;
//!
//! // Build from file
//! let graph = OnnxGraphBuilder::new().parse_file("model.onnx")?;
//!
//! // Build from bytes
//! let graph = OnnxGraphBuilder::new().parse_bytes(&bytes)?;
//!
//! // Build from reader
//! let graph = OnnxGraphBuilder::new().parse_reader(file)?;
//! ```

use std::io::Read;
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
pub enum Error {
    /// Failed to open or read the ONNX file
    Io { path: String, error: std::io::Error },

    /// Failed to parse ONNX protobuf format
    InvalidFormat { path: Option<String>, error: String },

    /// ONNX opset version is not supported
    UnsupportedOpset {
        found: usize,
        minimum_required: usize,
    },

    /// Model graph nodes are not topologically sorted (ONNX spec violation)
    InvalidGraphStructure { reason: String },

    /// Missing required opset version for default domain
    MissingOpsetVersion,

    /// Type inference failed during IR conversion
    TypeInference(ProcessError),

    /// Generic processing error
    Processing(ProcessError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io { path, error } => {
                write!(f, "Failed to open ONNX file '{}': {}", path, error)
            }
            Error::InvalidFormat { path, error } => {
                if let Some(p) = path {
                    write!(f, "Invalid ONNX format in '{}': {}", p, error)
                } else {
                    write!(f, "Invalid ONNX format: {}", error)
                }
            }
            Error::UnsupportedOpset {
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
            Error::InvalidGraphStructure { reason } => {
                write!(f, "Invalid ONNX graph structure: {}", reason)
            }
            Error::MissingOpsetVersion => {
                write!(
                    f,
                    "ONNX model must specify opset version for default domain"
                )
            }
            Error::TypeInference(e) => {
                write!(f, "Type inference failed: {:?}", e)
            }
            Error::Processing(e) => {
                write!(f, "Processing error: {:?}", e)
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io { error, .. } => Some(error),
            _ => None,
        }
    }
}

impl From<ProcessError> for Error {
    fn from(error: ProcessError) -> Self {
        Error::Processing(error)
    }
}

/// ONNX IR builder with fluent API
///
/// Builds ONNX IR graphs from various sources (files, bytes, readers).
/// Future configuration options can be added without breaking changes.
///
/// # Examples
///
/// ```ignore
/// use onnx_ir::OnnxGraphBuilder;
///
/// // Build from file (uses mmap when feature is enabled)
/// let graph = OnnxGraphBuilder::new().parse_file("model.onnx")?;
///
/// // Build from bytes
/// let graph = OnnxGraphBuilder::new().parse_bytes(&model_bytes)?;
///
/// // Build from reader
/// let graph = OnnxGraphBuilder::new().parse_reader(std::io::Cursor::new(data))?;
/// ```
#[derive(Debug, Clone, Default)]
pub struct OnnxGraphBuilder {
    // Future options can be added here without breaking changes
    // e.g., strict_mode: bool, min_opset_version: Option<usize>
}

impl OnnxGraphBuilder {
    /// Create a new ONNX graph builder with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse an ONNX model from a file path
    ///
    /// When the `mmap` feature is enabled (default), the file is memory-mapped
    /// for zero-copy tensor loading, significantly reducing memory usage.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - File cannot be opened or read
    /// - File is not valid ONNX protobuf format
    /// - ONNX opset version is less than 16
    /// - Graph nodes are not topologically sorted
    /// - Type inference fails
    pub fn parse_file(self, path: impl AsRef<Path>) -> Result<OnnxGraph, Error> {
        let path = path.as_ref();
        log::info!("Parsing ONNX file: {}", path.display());

        // Load file contents - mmap when feature is enabled
        #[cfg(feature = "mmap")]
        let buffer = {
            let file = File::open(path).map_err(|error| Error::Io {
                path: path.display().to_string(),
                error,
            })?;
            // SAFETY: We're mapping a read-only file. The bytes::Bytes keeps
            // the mmap alive for as long as tensor data references it.
            let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|error| Error::Io {
                path: path.display().to_string(),
                error,
            })?;
            log::debug!("Memory-mapped ONNX file ({} bytes)", mmap.len());
            bytes::Bytes::from_owner(mmap)
        };

        #[cfg(not(feature = "mmap"))]
        let buffer = {
            let mut file = File::open(path).map_err(|error| Error::Io {
                path: path.display().to_string(),
                error,
            })?;
            let mut buf = Vec::new();
            file.read_to_end(&mut buf).map_err(|error| Error::Io {
                path: path.display().to_string(),
                error,
            })?;
            log::debug!("Read ONNX file into memory ({} bytes)", buf.len());
            bytes::Bytes::from(buf)
        };

        self.parse_buffer(buffer, Some(path))
    }

    /// Parse an ONNX model from a byte slice
    ///
    /// Note: This copies the data internally. For large models already in memory
    /// as `bytes::Bytes`, consider using the internal buffer directly.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Data is not valid ONNX protobuf format
    /// - ONNX opset version is less than 16
    /// - Graph nodes are not topologically sorted
    /// - Type inference fails
    pub fn parse_bytes(self, data: &[u8]) -> Result<OnnxGraph, Error> {
        let buffer = bytes::Bytes::copy_from_slice(data);
        self.parse_buffer(buffer, None)
    }

    /// Parse an ONNX model from a reader
    ///
    /// Reads all data into memory before parsing.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Reading from the reader fails
    /// - Data is not valid ONNX protobuf format
    /// - ONNX opset version is less than 16
    /// - Graph nodes are not topologically sorted
    /// - Type inference fails
    pub fn parse_reader<R: Read>(self, mut reader: R) -> Result<OnnxGraph, Error> {
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).map_err(|error| Error::Io {
            path: "<reader>".to_string(),
            error,
        })?;
        log::debug!("Read ONNX from reader ({} bytes)", buf.len());
        let buffer = bytes::Bytes::from(buf);
        self.parse_buffer(buffer, None)
    }

    /// Internal: Parse from a bytes::Bytes buffer
    fn parse_buffer(
        self,
        buffer: bytes::Bytes,
        source_path: Option<&Path>,
    ) -> Result<OnnxGraph, Error> {
        let path_str = source_path.map(|p| p.display().to_string());

        let model: ModelProto =
            Message::parse_from_tokio_bytes(&buffer).map_err(|e| Error::InvalidFormat {
                path: path_str.clone(),
                error: e.to_string(),
            })?;

        if !verify_opsets(&model.opset_import, MIN_OPSET_VERSION) {
            let found_version = model
                .opset_import
                .iter()
                .find(|opset| opset.domain.is_empty())
                .map(|opset| opset.version as usize)
                .unwrap_or(0);

            return Err(Error::UnsupportedOpset {
                found: found_version,
                minimum_required: MIN_OPSET_VERSION,
            });
        }

        // ONNX nodes must be topologically sorted per spec:
        // https://github.com/onnx/onnx/blob/main/docs/IR.md#graphs
        if !model.graph.node.is_top_sorted() {
            return Err(Error::InvalidGraphStructure {
                reason: "Nodes are not topologically sorted (ONNX spec violation)".to_string(),
            });
        }

        let graph = build_graph(&model)?;

        if let Some(path) = path_str {
            log::info!("Finished parsing ONNX file: {}", path);
        } else {
            log::info!("Finished parsing ONNX from bytes");
        }
        Ok(graph)
    }
}

/// Build IR graph from ONNX model through 5 phases:
/// 1. Initialization 2. Node Conversion 3. Type Inference 4. Post-processing 5. Finalization
///
/// # Errors
///
/// Returns an error if:
/// - Missing opset version for default domain
/// - Type inference fails
pub fn build_graph(model: &ModelProto) -> Result<OnnxGraph, Error> {
    let opset_version = extract_opset_version(model)?;
    build_graph_from_proto(&model.graph, opset_version)
}

/// Build IR graph from ONNX GraphProto (for subgraphs)
///
/// # Errors
///
/// Returns an error if node conversion or type inference fails
pub fn build_graph_from_proto(
    graph: &crate::protos::GraphProto,
    opset_version: usize,
) -> Result<OnnxGraph, Error> {
    build_graph_from_proto_with_registry(graph, opset_version, None)
}

/// Build IR graph with shared name registry (for sibling subgraphs)
///
/// # Errors
///
/// Returns an error if node conversion or type inference fails
pub fn build_graph_from_proto_with_registry(
    graph: &crate::protos::GraphProto,
    opset_version: usize,
    name_registry: Option<crate::graph_state::NameRegistry>,
) -> Result<OnnxGraph, Error> {
    let graph_builder = build_graph_builder_from_proto(graph, opset_version, name_registry)?;

    log::debug!(" PHASE 6: Node Conversion (RawNode -> Node) ");
    Ok(graph_builder.convert_to_graph(opset_version))
}

/// Build IR graph as OnnxGraphBuilder (for subgraphs during processing)
///
/// This returns OnnxGraphBuilder which still contains RawNode instances.
/// Call convert_to_graph() to get the final OnnxGraph with Node enum instances.
///
/// # Errors
///
/// Returns an error if node conversion or type inference fails
pub(crate) fn build_graph_builder_from_proto(
    graph: &crate::protos::GraphProto,
    opset_version: usize,
    name_registry: Option<crate::graph_state::NameRegistry>,
) -> Result<crate::ir::OnnxGraphBuilder, Error> {
    build_graph_builder_from_proto_with_outer_scope(
        graph,
        opset_version,
        name_registry,
        crate::ir::OuterScopeTypes::new(),
    )
}

/// Build IR graph as OnnxGraphBuilder with access to outer scope types
///
/// This is used for building subgraphs that reference values from parent graphs.
/// The `outer_scope` map provides types for values that the subgraph references
/// but doesn't define internally.
///
/// # Errors
///
/// Returns an error if node conversion or type inference fails
pub(crate) fn build_graph_builder_from_proto_with_outer_scope(
    graph: &crate::protos::GraphProto,
    opset_version: usize,
    name_registry: Option<crate::graph_state::NameRegistry>,
    outer_scope: crate::ir::OuterScopeTypes,
) -> Result<crate::ir::OnnxGraphBuilder, Error> {
    log::debug!(" PHASE 1: Initialization ");
    let state_rc =
        initialization::initialize_from_graph_with_registry_and_outer_scope(graph, name_registry, outer_scope);

    log::debug!(" PHASE 2: Node Conversion (Proto -> RawNode) ");
    node_conversion::convert_nodes_from_graph(graph, &state_rc, opset_version)?;

    log::debug!(" PHASE 3: Type Inference ");
    type_inference::infer_types(&state_rc, opset_version).map_err(Error::TypeInference)?;

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
fn extract_opset_version(model: &ModelProto) -> Result<usize, Error> {
    model
        .opset_import
        .iter()
        .find(|opset| opset.domain.is_empty())
        .map(|opset| opset.version as usize)
        .ok_or(Error::MissingOpsetVersion)
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
