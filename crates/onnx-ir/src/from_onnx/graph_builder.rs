//! ONNX graph builder and parser
//!
//! This module contains the main logic for converting ONNX protobuf models
//! into the internal IR representation.
//!
//! MUST READ: Absolutely no node type specific logic

use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    fs::File,
    path::Path,
    rc::Rc,
};

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

        let graph_data = GraphData::new(
            &model_proto.graph.input,
            &model_proto.graph.output,
            &model_proto.graph.initializer,
        );

        // Wrap GraphData in Rc<RefCell<>> to allow shared mutable access
        let graph_data_rc = Rc::new(RefCell::new(graph_data));

        // Attach value_store to all initializer constant nodes
        // (These were created in GraphData::new() but couldn't have value_store set yet)
        // Also initialize node_name_counter to account for these pre-existing constants
        {
            let mut graph_data = graph_data_rc.borrow_mut();
            let mut constant_count = 0;
            for node in &mut graph_data.processed_nodes {
                if node.node_type == NodeType::Constant {
                    for arg in &mut node.inputs {
                        arg.value_store = Some(graph_data_rc.clone());
                    }
                    for arg in &mut node.outputs {
                        arg.value_store = Some(graph_data_rc.clone());
                    }
                    constant_count += 1;
                }
            }

            // Initialize the constant counter so subsequent ONNX Constant nodes don't collide
            if constant_count > 0 {
                self.node_name_counter
                    .insert(NodeType::Constant, constant_count);
                log::debug!(
                    "Initialized Constant node counter to {} (from initializers)",
                    constant_count
                );
            }
        }

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

            // For Constant nodes: move tensor data from attributes to central store
            if node.node_type == NodeType::Constant {
                use crate::ir::{AttributeValue, TensorData};

                // Find the value attribute (could be "value", "value_float", "value_floats", etc.)
                let keys = [
                    "value",
                    "value_float",
                    "value_floats",
                    "value_int",
                    "value_ints",
                    "value_string",
                    "value_strings",
                ];

                if let Some(attr_key) = keys.iter().find(|&key| node.attrs.contains_key(*key))
                    && let Some(attr_value) = node.attrs.get(*attr_key)
                {
                    // Convert attribute to TensorData if possible
                    let tensor_data_opt: Option<TensorData> = match attr_value {
                        AttributeValue::Tensor(tensor) => Some(tensor.clone()),
                        AttributeValue::Float32(val) => Some(TensorData {
                            shape: vec![],
                            data: crate::ir::Data::Float32(*val),
                        }),
                        AttributeValue::Float32s(vals) => Some(TensorData {
                            shape: vec![vals.len()],
                            data: crate::ir::Data::Float32s(vals.clone()),
                        }),
                        AttributeValue::Int64(val) => Some(TensorData {
                            shape: vec![],
                            data: crate::ir::Data::Int64(*val),
                        }),
                        AttributeValue::Int64s(vals) => Some(TensorData {
                            shape: vec![vals.len()],
                            data: crate::ir::Data::Int64s(vals.clone()),
                        }),
                        _ => None,
                    };

                    if let Some(tensor_data) = tensor_data_opt {
                        // Allocate ID and store data in central store
                        let data_id = {
                            let mut graph_data = graph_data_rc.borrow_mut();
                            graph_data.store_tensor_data(tensor_data.clone())
                        };

                        // Create type based on tensor data
                        let ty = if tensor_data.shape.is_empty() {
                            crate::ir::ArgType::Scalar(tensor_data.elem_type())
                        } else {
                            crate::ir::ArgType::Tensor(crate::ir::TensorType {
                                elem_type: tensor_data.elem_type(),
                                rank: tensor_data.shape.len(),
                                static_shape: Some(tensor_data.shape.clone()),
                            })
                        };

                        // Create input with Static value
                        node.inputs.push(crate::ir::Argument {
                            name: String::new(), // Empty name for internal static input
                            ty: ty.clone(),
                            data_id: Some(data_id),
                            value_source: crate::ir::ValueSource::Static,
                            value_store: Some(graph_data_rc.clone()),
                        });

                        // Set type and value_source on output
                        // We set the type now (before other nodes reference it) because type inference
                        // happens after all nodes are created, but inputs are cloned when nodes are created
                        if !node.outputs.is_empty() {
                            node.outputs[0].value_source = crate::ir::ValueSource::Constant;
                            node.outputs[0].ty = ty;
                        }

                        // Remove tensor data from attributes
                        node.attrs.remove(*attr_key);
                    }
                }
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

            // NOTE: potential start of custom functions
            // can filter, coalesce, or modify the nodes here
            // args : node, peek_iter, graph_data

            log::debug!("Processing node: {}", node.name);
            let registry = get_processor_registry();
            let processor = registry.get(&node.node_type);

            // Lift constants by converting Constant arguments to Static
            // This embeds constant data directly in the argument via to_static()
            processor
                .lift_constants(&mut node, opset_version)
                .unwrap_or_else(|e| {
                    panic!(
                        "Failed to lift constants for node {} (type: {:?}): {:?}",
                        node.name, node.node_type, e
                    )
                });

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

        // Extract the processed graph data while preserving tensor_data for .value() access
        let (mut processed_nodes, inputs, mut outputs) = {
            let mut graph_data = graph_data_rc.borrow_mut();

            // Clone tensor_data before consuming (we need to keep it in GraphData for .value())
            let tensor_data_clone = graph_data.tensor_data.clone();
            let next_tensor_id = graph_data.next_tensor_id;

            // Consume to get nodes/inputs/outputs
            let result =
                std::mem::replace(&mut *graph_data, GraphData::new(&[], &[], &[])).consume();

            // Restore tensor_data so .value() still works for burn-import
            // This allows Arguments to access their data via data_id
            graph_data.tensor_data = tensor_data_clone;
            graph_data.next_tensor_id = next_tensor_id;

            result
        };

        // Eliminate Identity nodes BEFORE filtering out constants
        // This ensures Identity rewiring can copy data_id from constant outputs
        // 1. Identity->Constant conversion already happened during node processing
        // 2. Now remove pass-through Identity nodes and rewire connections
        // 3. Preserve at least one Identity if graph would be empty
        log::debug!("Starting Identity elimination");
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

        // Re-run constant lifting after Identity elimination
        // Identity nodes may have been rewired to constant outputs, which now need to be lifted
        log::debug!("Re-running constant lifting after Identity elimination");
        {
            // Rebuild graph_data state with post-Identity-elimination nodes
            let mut graph_data = graph_data_rc.borrow_mut();
            graph_data.processed_nodes = processed_nodes.clone();

            // Rebuild constant_nodes map with current indices
            let mut new_constant_nodes = std::collections::HashMap::new();
            for (idx, n) in processed_nodes.iter().enumerate() {
                if n.node_type == NodeType::Constant {
                    for output in &n.outputs {
                        new_constant_nodes.insert(output.name.clone(), idx);
                    }
                }
            }
            graph_data.constant_nodes = new_constant_nodes;
            drop(graph_data);

            // Re-attach value_store to all node inputs
            for node in &mut processed_nodes {
                for arg in &mut node.inputs {
                    arg.value_store = Some(graph_data_rc.clone());
                }
            }

            // Now try to lift constants for each node
            let registry = get_processor_registry();
            for node in &mut processed_nodes {
                let processor = registry.get(&node.node_type);
                processor
                    .lift_constants(node, opset_version)
                    .unwrap_or_else(|e| {
                        // It's OK if this fails - some inputs may already be Static from first pass
                        log::debug!(
                            "Could not lift constants for node {} after Identity elimination (this is normal): {:?}",
                            node.name,
                            e
                        );
                    });
            }
        }

        // Remove unreferenced constant nodes
        // After constant lifting via to_static(), constants may have no remaining runtime references.
        // We count references based on ValueSource:
        // - Static: value embedded, doesn't reference the constant node (lifted)
        // - Constant: points to constant node output (not lifted, keep the constant)
        // - Dynamic: points to runtime node output (not a constant reference)
        // Remove constant nodes that have zero Constant/Dynamic references.
        log::debug!("Counting references to constant nodes");

        // Build a map of constant output names to node indices
        let mut constant_output_to_idx: HashMap<String, usize> = HashMap::new();
        for (idx, node) in processed_nodes.iter().enumerate() {
            if node.node_type == NodeType::Constant {
                for output in &node.outputs {
                    constant_output_to_idx.insert(output.name.clone(), idx);
                }
            }
        }

        // Count references: output_name -> reference_count
        let mut constant_references: HashMap<String, usize> = HashMap::new();

        // Count references from node inputs
        for node in &processed_nodes {
            for input in &node.inputs {
                // Only count Constant and Dynamic references (not Static)
                // Static inputs have empty names and embedded values via data_id,
                // so they don't reference the constant node
                if (input.is_constant() || input.is_dynamic())
                    && constant_output_to_idx.contains_key(&input.name)
                {
                    *constant_references.entry(input.name.clone()).or_insert(0) += 1;
                }
            }
        }

        // Count references from graph outputs
        for output in &outputs {
            if (output.is_constant() || output.is_dynamic())
                && constant_output_to_idx.contains_key(&output.name)
            {
                *constant_references.entry(output.name.clone()).or_insert(0) += 1;
            }
        }

        // Mark constant nodes with zero references for removal
        let mut constants_to_remove = HashSet::new();
        for (output_name, &node_idx) in &constant_output_to_idx {
            let ref_count = constant_references.get(output_name).unwrap_or(&0);
            if *ref_count == 0 {
                constants_to_remove.insert(node_idx);
                log::debug!(
                    "Marking constant node at index {} (output: {}) for removal (no references)",
                    node_idx,
                    output_name
                );
            }
        }

        // Filter out unreferenced constant nodes
        log::debug!(
            "Filtering nodes: total={}, nodes_to_remove={:?}",
            processed_nodes.len(),
            constants_to_remove
        );

        let mut i = 0;
        processed_nodes.retain(|node| {
            let keep = !constants_to_remove.contains(&i);

            if !keep {
                log::debug!("Filtering out node at index {}: {}", i, node.name);
            }
            i += 1;
            keep
        });
        log::debug!("After filtering: {} nodes remain", processed_nodes.len());

        // TODO Update graph inputs and outputs to match the processed nodes inputs and outputs
        // This is necessary for the graph to be valid
        // ConstantOfShape updates input to be Shape argument and output Tensor dim is updated
        OnnxGraph {
            nodes: processed_nodes,
            inputs,
            outputs,
            _graph_data: Some(graph_data_rc),
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
