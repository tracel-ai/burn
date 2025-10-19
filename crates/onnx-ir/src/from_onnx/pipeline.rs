//! ONNX to IR conversion pipeline orchestrator
//!
//! This module provides the high-level orchestration of the ONNX conversion process.
//! It clearly shows the entire conversion flow from start to finish.

use std::{fs::File, path::Path};

use protobuf::Message;

use crate::{ir::OnnxGraph, protos::ModelProto, util::verify_opsets};

use super::{
    conversion::MIN_OPSET_VERSION,
    phases::{finalization, initialization, node_conversion, post_processing, type_inference},
};

/// Main entry point: Parse an ONNX file and convert to IR
///
/// # Arguments
///
/// * `onnx_path` - Path to the ONNX model file
///
/// # Returns
///
/// * `OnnxGraph` - The internal graph representation
///
/// # Panics
///
/// * If the file cannot be opened or parsed
/// * If the model uses an unsupported opset version
/// * If nodes are not topologically sorted
pub fn parse_onnx(onnx_path: &Path) -> OnnxGraph {
    log::info!("Parsing ONNX file: {}", onnx_path.display());

    // Load and validate model
    let mut file = File::open(onnx_path)
        .unwrap_or_else(|_| panic!("Unable to open file: {}", onnx_path.display()));
    let model: ModelProto =
        Message::parse_from_reader(&mut file).expect("Unable to parse ONNX file");

    if !verify_opsets(&model.opset_import, MIN_OPSET_VERSION) {
        panic!(
            "Unsupported ONNX opset version. Requires opset {MIN_OPSET_VERSION} or higher. \
            See documentation for upgrade instructions."
        );
    }

    // ONNX nodes must be topologically sorted per spec:
    // https://github.com/onnx/onnx/blob/main/docs/IR.md#graphs
    debug_assert!(
        model.graph.node.is_top_sorted(),
        "Nodes are not topologically sorted"
    );

    log::debug!("Number of nodes: {}", model.graph.node.len());
    log::debug!("Number of inputs: {}", model.graph.input.len());
    log::debug!("Number of initializers: {}", model.graph.initializer.len());
    log::debug!("Number of outputs: {}", model.graph.output.len());

    let graph = build_graph(&model);

    log::info!("Finished parsing ONNX file: {}", onnx_path.display());
    graph
}

/// Build an IR graph from an ONNX model
///
/// This is the main conversion pipeline with clearly separated phases:
///
/// 1. **Initialization** - Create state, process initializers
/// 2. **Node Conversion** - Convert ONNX nodes, extract constants, coalesce
/// 3. **Type Inference** - Infer types iteratively with preferences
/// 4. **Post-processing** - Identity elimination, constant lifting
/// 5. **Finalization** - Remove unused constants, build final graph
pub fn build_graph(model: &ModelProto) -> OnnxGraph {
    log::debug!(" PHASE 1: Initialization ");
    let state_rc = initialization::initialize(model);

    log::debug!(" PHASE 2: Node Conversion ");
    node_conversion::convert_nodes(model, &state_rc);

    log::debug!(" PHASE 3: Type Inference ");
    let opset_version = extract_opset_version(model);
    type_inference::infer_types(&state_rc, opset_version);

    log::debug!(" PHASE 4: Post-processing ");
    let (mut nodes, inputs, mut outputs) = post_processing::post_process(&state_rc);

    log::debug!(" PHASE 5: Finalization ");
    finalization::finalize(&mut nodes, inputs, &mut outputs, state_rc)
}

/// Extract opset version from model (default ONNX domain)
fn extract_opset_version(model: &ModelProto) -> usize {
    model
        .opset_import
        .iter()
        .find(|opset| opset.domain.is_empty())
        .map(|opset| opset.version as usize)
        .unwrap_or(MIN_OPSET_VERSION as usize)
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
