//! ONNX graph builder and parser
//!
//! This module contains the main logic for converting ONNX protobuf models
//! into the internal IR representation.

use std::{cell::RefCell, collections::HashMap, fs::File, path::Path, rc::Rc};

use protobuf::{Enum, Message};

use crate::{
    coalesce::coalesce,
    ir::{Node, NodeType, OnnxGraph},
    node_remap::remap_node_type,
    proto_conversion::convert_node_proto,
    protos::ModelProto,
    util::verify_opsets,
};

use super::{
    conversion::{MIN_OPSET_VERSION, TopologicalSortable},
    get_processor_registry,
    graph_data::GraphData,
    type_inference::iterative_type_inference_with_preferences,
};

/// Builder for constructing ONNX graphs
#[derive(Default)]
pub(crate) struct OnnxGraphBuilder {
    node_name_counter: HashMap<NodeType, usize>,
}

impl OnnxGraphBuilder {
    /// Build an OnnxGraph from a ModelProto
    ///
    /// This is the main entry point for ONNX conversion. It performs:
    /// 1. Initialization from proto structures
    /// 2. Node processing, coalescing, and transformation
    /// 3. Constant lifting and value caching
    /// 4. Iterative type inference with preference propagation
    /// 5. Graph finalization and cleanup
    pub(crate) fn build(mut self, model_proto: &ModelProto) -> OnnxGraph {
        // Extract opset version from model (default ONNX domain)
        let opset_version = model_proto
            .opset_import
            .iter()
            .find(|opset| opset.domain.is_empty())
            .map(|opset| opset.version as usize)
            .unwrap_or(MIN_OPSET_VERSION as usize);

        // Initialize Constant node counter to account for initializers
        let num_initializers = model_proto.graph.initializer.len();
        if num_initializers > 0 {
            self.node_name_counter
                .insert(NodeType::Constant, num_initializers);
        }

        let graph_data = GraphData::new(
            &model_proto.graph.input,
            &model_proto.graph.output,
            &model_proto.graph.initializer,
        );

        // Wrap GraphData in Rc<RefCell<>> to allow shared mutable access
        let graph_data_rc = Rc::new(RefCell::new(graph_data));

        for t in &model_proto.graph.initializer {
            log::debug!(
                "init name={:?} dtype={:?} dims={:?} raw_len={} i32={} i64={} f32={} f64={}",
                t.name,
                crate::protos::tensor_proto::DataType::from_i32(t.data_type),
                t.dims,
                t.raw_data.len(),
                t.int32_data.len(),
                t.int64_data.len(),
                t.float_data.len(),
                t.double_data.len(),
            );
        }

        let mut node_iter = model_proto.graph.node.iter().peekable();

        // Process nodes
        while let Some(node_proto) = node_iter.next() {
            let mut node = convert_node_proto(node_proto, &graph_data_rc.borrow());

            // Attach value_store to all arguments in the node
            for arg in &mut node.inputs {
                arg.value_store = Some(graph_data_rc.clone());
            }
            for arg in &mut node.outputs {
                arg.value_store = Some(graph_data_rc.clone());
            }

            remap_node_type(&mut node);
            self.handle_node_renaming(&mut node);

            // Track node type before coalesce
            let node_type_before_coalesce = node.node_type.clone();

            coalesce(&mut node, &mut node_iter, &mut graph_data_rc.borrow_mut());

            // Re-attach value_stores after coalesce (which may add new inputs from fusion)
            for arg in &mut node.inputs {
                arg.value_store = Some(graph_data_rc.clone());
            }
            for arg in &mut node.outputs {
                arg.value_store = Some(graph_data_rc.clone());
            }

            // If coalesce changed the node type (e.g., Gemm->Linear, MatMul->Linear), rename it
            if node.node_type != node_type_before_coalesce {
                self.handle_node_renaming(&mut node);
            }

            // Convert Identity nodes with constant inputs to Constant nodes
            // This allows burn-import to access the constant values via into_value()
            if node.node_type == NodeType::Identity && !node.inputs.is_empty() {
                let input_name = &node.inputs[0].name;
                let has_constant_input = {
                    let graph_data = graph_data_rc.borrow();
                    graph_data.is_constant(input_name)
                };

                if has_constant_input {
                    // Convert Identity to Constant node
                    let constant_value = {
                        let graph_data = graph_data_rc.borrow();
                        graph_data.get_value(input_name)
                    };

                    if let Some(tensor_data) = constant_value {
                        log::debug!(
                            "Converting Identity node {} to Constant (input: {})",
                            node.name,
                            input_name
                        );

                        node.node_type = NodeType::Constant;
                        node.attrs.insert(
                            "value".to_string(),
                            crate::ir::AttributeValue::Tensor(tensor_data),
                        );
                        node.inputs.clear(); // Constant nodes have no inputs

                        // Rename since we changed type
                        self.handle_node_renaming(&mut node);

                        // Re-attach value_stores after renaming
                        for arg in &mut node.outputs {
                            arg.value_store = Some(graph_data_rc.clone());
                        }
                    }
                }
            }

            // NOTE: potential start of custom functions
            // can filter, coalesce, or modify the nodes here
            // args : node, peek_iter, graph_data

            log::debug!("Processing node: {}", node.name);
            let registry = get_processor_registry();
            let processor = registry.get(&node.node_type);

            // Register ALL Constant nodes so their values can be accessed via has_value() and get_value()
            // This includes: initializer constants (already registered), converted Identity nodes,
            // and explicit ONNX Constant nodes
            if node.node_type == NodeType::Constant && !node.outputs.is_empty() {
                let future_output_name = format!("{}_out1", node.name);
                let node_idx = {
                    let graph_data = graph_data_rc.borrow();
                    graph_data.get_current_index()
                };

                // Only register if not already registered (e.g., initializer constants)
                {
                    let mut graph_data = graph_data_rc.borrow_mut();
                    if !graph_data.constant_nodes.contains_key(&future_output_name) {
                        graph_data
                            .constant_nodes
                            .insert(future_output_name.clone(), node_idx);
                    }
                } // Explicitly drop mutable borrow here
            }

            // Lift constants (ensure constant inputs are accessible)
            // lift_constants returns a list of input names that COULD be lifted
            // We filter by has_value() to only lift actual constants
            let potential_lifts = processor
                .lift_constants(&mut node, opset_version)
                .unwrap_or_else(|e| {
                    panic!(
                        "Failed to lift constants for node {} (type: {:?}): {:?}",
                        node.name, node.node_type, e
                    )
                });

            // Filter to only lift inputs that are constants (have values available)
            // All constants are liftable - initializers, ONNX Constant nodes, etc.
            // Check GraphData directly to avoid RefCell borrow conflicts
            let lifted: Vec<String> = {
                let graph_data = graph_data_rc.borrow();
                potential_lifts
                    .into_iter()
                    .filter(|input_name| graph_data.has_value(input_name))
                    .collect()
            }; // Drop immutable borrow here

            // Make lifted constants accessible by caching their values
            // Identity nodes with constant values have already been converted to Constant nodes
            for input_name in &lifted {
                {
                    let mut graph_data = graph_data_rc.borrow_mut();

                    // Get the value from the constant and cache it
                    if let Some(value) = graph_data.get_value(input_name) {
                        // Cache the value to ensure it stays available for burn-import
                        graph_data.consumed_values.insert(input_name.clone(), value);
                        log::debug!("Lifted constant {} for node {}", input_name, node.name);

                        // Only remove constants that are ALWAYS embedded statically in configs
                        // Conv/Linear weights are fully embedded and never referenced in forward()
                        // Other node types (Reshape, Slice, etc.) may use Runtime constants
                        let should_remove = matches!(
                            node.node_type,
                            NodeType::Conv1d
                                | NodeType::Conv2d
                                | NodeType::Conv3d
                                | NodeType::ConvTranspose1d
                                | NodeType::ConvTranspose2d
                                | NodeType::ConvTranspose3d
                                | NodeType::Linear
                        );

                        if should_remove
                            && let Some(&const_node_idx) = graph_data.constant_nodes.get(input_name)
                        {
                            graph_data.nodes_to_remove.insert(const_node_idx);
                            log::debug!(
                                "Marked constant node at index {} for removal (fully embedded in {} config)",
                                const_node_idx,
                                node.name
                            );
                        }
                    } else {
                        log::warn!(
                            "Failed to lift constant {} for node {} - value not found",
                            input_name,
                            node.name
                        );
                    }
                } // Explicitly drop mutable borrow before next iteration
            }

            // Extract config first
            let config = processor
                .extract_config(&node, opset_version)
                .unwrap_or_else(|e| {
                    panic!(
                        "Failed to extract config for node {} (type: {:?}): {:?}",
                        node.name, node.node_type, e
                    )
                });
            node.config = config;

            // Add node to graph (type inference happens later in iterative loop)
            graph_data_rc.borrow_mut().add_node(node);
        }

        // Run iterative type inference with preference propagation
        // This allows preferences to be collected based on inferred types,
        // enabling scenarios like Concat requesting Shape types after seeing Shape inputs
        log::debug!("Starting iterative type inference with preference propagation");
        {
            // Temporarily extract nodes to avoid holding mutable borrow during iteration
            // (iteration may need immutable borrows for into_value() calls)
            let mut nodes = std::mem::take(&mut graph_data_rc.borrow_mut().processed_nodes);
            iterative_type_inference_with_preferences(&mut nodes, opset_version);
            graph_data_rc.borrow_mut().processed_nodes = nodes;
        }

        // Cache all Constant node values for burn-import to access
        // This ensures burn-import can generate code for ALL constants, not just lifted ones
        {
            let mut graph_data = graph_data_rc.borrow_mut();

            // Collect constant nodes that need caching (to avoid borrow issues)
            let constant_outputs: Vec<String> = graph_data
                .processed_nodes
                .iter()
                .filter(|node| node.node_type == NodeType::Constant && !node.outputs.is_empty())
                .map(|node| node.outputs[0].name.clone())
                .collect();

            // Cache values for all constant nodes
            for output_name in constant_outputs {
                if !graph_data.consumed_values.contains_key(&output_name)
                    && let Some(value) = graph_data.get_value(&output_name)
                {
                    graph_data
                        .consumed_values
                        .insert(output_name.clone(), value);
                    log::debug!("Cached constant {} value for burn-import", output_name);
                }
            }
        }

        // Extract the processed graph data and preserve consumed_values for burn-import
        let (mut processed_nodes, inputs, outputs, nodes_to_remove) = {
            let mut graph_data = graph_data_rc.borrow_mut();

            // Extract consumed_values before consuming
            let consumed_values = std::mem::take(&mut graph_data.consumed_values);

            // Consume the old graph_data
            let result =
                std::mem::replace(&mut *graph_data, GraphData::new(&[], &[], &[])).consume();

            // Restore consumed_values so burn-import can access them via into_value()
            graph_data.consumed_values = consumed_values;

            (result.0, result.1, result.2, result.3)
        };

        // Filter out nodes marked for removal
        //
        // Lifted constants whose values are fully embedded in node configs (e.g., Conv1d weights
        // serialized in Conv1dRecord, Linear weights, etc.) are marked for removal during the
        // lifting process. Their values remain accessible via consumed_values for burn-import
        // to generate serialized weights, but they don't need to exist as separate Constant nodes.
        //
        // Constants that are still needed at runtime (e.g., shape constants accessed during
        // execution) are not marked for removal and will appear in the final graph.
        log::debug!(
            "Filtering nodes: total={}, nodes_to_remove={:?}",
            processed_nodes.len(),
            nodes_to_remove
        );

        let mut i = 0;
        processed_nodes.retain(|node| {
            let keep = !nodes_to_remove.contains(&i);

            if !keep {
                log::debug!("Filtering out node at index {}: {}", i, node.name);
            }
            i += 1;
            keep
        });

        log::debug!("After filtering: {} nodes remain", processed_nodes.len());

        // Eliminate Identity nodes
        // 1. Identity->Constant conversion already happened during node processing
        // 2. Now remove pass-through Identity nodes and rewire connections
        // 3. Preserve at least one Identity if graph would be empty
        log::debug!("Starting Identity elimination");
        let mut outputs = outputs; // Make mutable for elimination
        {
            let elimination_plan = super::identity_elimination::plan_identity_elimination(
                &processed_nodes,
                &inputs,
                &outputs,
            );
            super::identity_elimination::apply_identity_elimination(
                &mut processed_nodes,
                &mut outputs,
                elimination_plan,
            );
        }
        log::debug!(
            "After Identity elimination: {} nodes remain",
            processed_nodes.len()
        );

        // TODO Update graph inputs and outputs to match the processed nodes inputs and outputs
        // This is necessary for the graph to be valid
        // ConstantOfShape updates input to be Shape argument and output Tensor dim is updated
        OnnxGraph {
            nodes: processed_nodes,
            inputs,
            outputs,
        }
    }

    /// Handle node renaming based on node type counters
    fn handle_node_renaming(&mut self, node: &mut Node) {
        self.node_name_counter
            .entry(node.node_type.clone())
            .and_modify(|e| *e += 1)
            .or_insert(1);
        let new_name = format!(
            "{}{}",
            node.node_type, self.node_name_counter[&node.node_type]
        )
        .to_lowercase();

        log::debug!("Renaming node {:?} to {new_name:?}", &node.name);

        node.name.clone_from(&new_name);
    }
}

/// Parses an ONNX model file and converts it to an intermediate representation.
///
/// This function reads an ONNX model from the specified path, validates its opset version,
/// and transforms it into our internal graph representation for further processing.
///
/// # Arguments
///
/// * `onnx_path` - Path to the ONNX model file
///
/// # Returns
///
/// * `OnnxGraph` - The internal graph representation of the ONNX model
///
/// # Panics
///
/// * If the file cannot be opened or read
/// * If the ONNX model cannot be parsed
/// * If the model uses an unsupported opset version (must be >= MIN_OPSET_VERSION)
/// * If the nodes in the graph are not topologically sorted
pub fn parse_onnx(onnx_path: &Path) -> OnnxGraph {
    log::info!("Parsing ONNX file: {}", onnx_path.display());

    // Open the file
    let mut file = File::open(onnx_path)
        .unwrap_or_else(|_| panic!("Unable to open file: {}", onnx_path.display()));
    let onnx_model: ModelProto =
        Message::parse_from_reader(&mut file).expect("Unable to parse ONNX file");

    // Check opset versions - must be >= MIN_OPSET_VERSION
    if !verify_opsets(&onnx_model.opset_import, MIN_OPSET_VERSION) {
        panic!(
            "Unsupported ONNX opset version. This implementation requires opset {MIN_OPSET_VERSION} or higher. \
            Please upgrade your model using the ONNX shape inference tool. \
            See documentation (https://burn.dev/books/burn/import/onnx-model.html) for details."
        );
    }

    // ONNX nodes must be topologically sorted per spec:
    // https://github.com/onnx/onnx/blob/main/docs/IR.md#graphs
    debug_assert!(
        onnx_model.graph.node.is_top_sorted(),
        "Nodes are not topologically sorted"
    );
    log::debug!("Number of nodes: {:?}", onnx_model.graph.node.len());
    log::debug!("Number of inputs: {:?}", onnx_model.graph.input.len());

    log::debug!(
        "Number of initializers: {:?}",
        onnx_model.graph.initializer.len()
    );

    log::debug!("Number of outputs: {:?}", onnx_model.graph.output.len());

    // Debug information about opset versions
    for opset in &onnx_model.opset_import {
        log::debug!(
            "Opset domain: {:?}, version: {:?}",
            if opset.domain.is_empty() {
                "<default>"
            } else {
                &opset.domain
            },
            opset.version
        );
    }

    let builder = OnnxGraphBuilder::default();
    let graph = builder.build(&onnx_model);

    log::info!("Finished parsing ONNX file: {}", onnx_path.display());

    graph
}
