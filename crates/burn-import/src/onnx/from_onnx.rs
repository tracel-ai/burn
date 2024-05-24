use std::{
    collections::{HashMap, HashSet},
    fs::File,
    path::Path,
};

use crate::onnx::{node_remap::remap_node_type, proto_conversion::convert_node_proto};

use super::{
    coalesce::coalesce,
    ir::{Attributes, Data, OnnxGraph, TensorType},
    protos::{ModelProto, NodeProto, TensorProto, ValueInfoProto},
};

use super::dim_inference::dim_inference;
use super::ir::{ArgType, Argument, Node, NodeType};

use protobuf::Message;

const LIFT_CONSTANTS_FOR_NODE_TYPES: [NodeType; 9] = [
    NodeType::BatchNormalization,
    NodeType::Clip,
    NodeType::Conv1d,
    NodeType::Conv2d,
    NodeType::Dropout,
    NodeType::Reshape,
    NodeType::Unsqueeze,
    NodeType::ReduceSum,
    NodeType::Squeeze,
];

#[derive(Debug, Clone)]
pub(crate) enum IOEntry {
    In(usize),
    Out(usize),
    Node(usize),
}

#[derive(Default, Debug, Clone)]
pub struct OnnxGraphIO {
    /// The inputs for the Graph
    pub(crate) inputs: Vec<Argument>,
    /// The outputs for the Graph
    pub(crate) outputs: Vec<Argument>,
    /// Initializers
    pub(crate) initializers: HashMap<String, Argument>,
    ///updated names of outputs of node not stored in the graph
    node_out: Vec<Argument>,
    pub(crate) old_io_names: HashMap<String, IOEntry>,
}

impl OnnxGraphIO {
    pub(crate) fn new(
        inputs: &Vec<ValueInfoProto>,
        outputs: &Vec<ValueInfoProto>,
        initializers: &Vec<TensorProto>,
    ) -> Self {
        let mut old_io_names = HashMap::new();
        let mut in_count = 1;
        let constants = initializers
            .iter()
            .map(|x| (x.name.clone(), Argument::from_initializer(x)))
            .collect::<HashMap<String, Argument>>();

        let inputs = inputs
            .iter()
            .enumerate()
            .map(|(i, x)| {
                let in_name = format!("input{}", in_count);
                old_io_names.insert(x.name.clone(), IOEntry::In(i));
                let mut arg = Argument::try_from(x.clone()).unwrap();
                if let Some(initial_arg) = constants.get(&x.name) {
                    if arg.value.is_none() {
                        arg.copy_value(initial_arg);
                    }
                }

                in_count += 1;
                arg.name = in_name;
                arg
            })
            .collect::<Vec<Argument>>();

        let outputs = outputs
            .iter()
            .enumerate()
            .map(|(i, x)| {
                old_io_names.insert(x.name.clone(), IOEntry::Out(i));
                Argument::try_from(x.clone()).unwrap()
            })
            .collect::<Vec<Argument>>();

        Self {
            inputs,
            outputs,
            initializers: constants,
            node_out: Vec::new(),
            old_io_names,
        }
    }

    pub fn update_output_name(&mut self, old_arg_name: &str, new_name: &str) {
        log::debug!(
            "old output name: {}\nnew output name: {}",
            &old_arg_name,
            &new_name
        );
        match self.old_io_names.get(old_arg_name) {
            Some(IOEntry::In(_)) => {
                panic!("input names are set from the beginning");
            }
            Some(IOEntry::Out(i)) => {
                let arg = self.outputs.get_mut(*i).unwrap();
                arg.name = new_name.to_string();
                //if you are updating the name of a graph output, it should be marked as passed
                arg.passed = true;
            }
            Some(IOEntry::Node(i)) => {
                let arg = self.node_out.get_mut(*i).unwrap();
                arg.name = new_name.to_string();
            }
            None => {
                //Constants, Casts wound up here before API changes
                panic!(
                    "Tried to update the name of {} to {} but entry doesn't exist in the map",
                    old_arg_name, new_name
                )
            }
        }
    }

    /// right now used for just remap unsqueeze to reshape
    pub fn add_generated_const(&mut self, name: &str, arg: Argument) {
        let idx = self.node_out.len();
        self.old_io_names
            .insert(name.to_string(), IOEntry::Node(idx));
        self.node_out.push(arg);
    }

    pub fn set_passed(&mut self, name: &str) {
        match self.old_io_names.get(name) {
            Some(IOEntry::In(i)) => {
                self.inputs[*i].passed = true;
            }
            Some(IOEntry::Out(i)) => {
                self.outputs[*i].passed = true;
            }
            Some(IOEntry::Node(i)) => {
                self.node_out[*i].passed = true;
            }
            None => panic!("No entry for {}", name),
        }
    }

    /// Used to initialize the input arguments for nodes. Names need to remain the same because
    /// currently the old names are the key for accessing the Argument
    pub fn init_in(&mut self, proto_str: &str) {
        match self.old_io_names.get(proto_str) {
            None => {
                self.old_io_names
                    .insert(proto_str.to_owned(), IOEntry::Node(self.node_out.len()));
                //NOTE: if initializers are guaranteed to be unique, (I think they are
                //need to confirm) then we could pop the initializer from the map
                if let Some(init_arg) = self.initializers.get(proto_str) {
                    self.node_out.push(init_arg.clone());
                } else {
                    //should this panic if outputs are initialized separately?
                    self.node_out.push(Argument::new(proto_str.to_owned()));
                }
            }
            Some(IOEntry::In(_)) | Some(IOEntry::Node(_)) => {}
            Some(IOEntry::Out(_)) => {
                panic!("graph output {} can't be a Node input", &proto_str)
            }
        }
    }

    pub fn init_out(&mut self, proto_str: &str) {
        match self.old_io_names.get(proto_str) {
            None => {
                self.old_io_names
                    .insert(proto_str.to_owned(), IOEntry::Node(self.node_out.len()));
                self.node_out.push(Argument::new(proto_str.to_owned()));
            }
            //already handled on graph_io creation
            Some(IOEntry::Out(_)) => {}
            Some(IOEntry::In(_)) => {
                panic!("graph input {} can't be a Node output", &proto_str)
            }
            Some(IOEntry::Node(i)) => {
                //TODO: this should panic if we rework coalesce to either not initialize the peeked node
                //or to process the node if it's not going to be removed. Currently, this panic would trigger
                //if we peeked a node, didn't coalesce it, and then processed it the next iteration
                //thus, the current best option is to just throw a warning
                log::warn!(
                    "output with old name {} was already generated by another node: {:?} ",
                    proto_str,
                    &self.node_out[*i]
                )
            }
        }
    }

    pub(crate) fn get_type(&self, name: &str) -> &ArgType {
        match self.old_io_names.get(name) {
            Some(IOEntry::In(i)) => &self.inputs[*i].ty,
            Some(IOEntry::Out(i)) => &self.outputs[*i].ty,
            Some(IOEntry::Node(i)) => &self.node_out[*i].ty,
            None => panic!("No entry for {}", name),
        }
    }

    /// Copy the type from one IO entry to another
    pub(crate) fn copy_type(&mut self, from_name: &str, to_name: &str) {
        let ty = match self.old_io_names.get(from_name) {
            Some(IOEntry::In(i)) => self.inputs[*i].ty.clone(),
            Some(IOEntry::Out(i)) => self.outputs[*i].ty.clone(),
            Some(IOEntry::Node(i)) => self.node_out[*i].ty.clone(),
            None => panic!("No entry {} to copy from", from_name),
        };

        match self.old_io_names.get(to_name) {
            Some(IOEntry::In(i)) => {
                self.inputs[*i].ty = ty;
            }
            Some(IOEntry::Out(i)) => {
                self.outputs[*i].ty = ty;
            }
            Some(IOEntry::Node(i)) => {
                self.node_out[*i].ty = ty;
            }
            None => panic!("No entry for {} to copy to", to_name),
        }
    }

    /// Copy value and type from one IO entry to another
    pub fn copy_value_type(&mut self, from_name: &str, to_name: &str) {
        let value = self.get_value(from_name).cloned();
        let ty = self.get_type(from_name).clone();

        match self.old_io_names.get(to_name) {
            Some(IOEntry::In(i)) => {
                self.inputs[*i].value = value;
                self.inputs[*i].ty = ty;
            }
            Some(IOEntry::Out(i)) => {
                self.outputs[*i].value = value;
                self.outputs[*i].ty = ty;
            }
            Some(IOEntry::Node(i)) => {
                self.node_out[*i].value = value;
                self.node_out[*i].ty = ty;
            }
            None => panic!("No entry for {} to copy to", to_name),
        }
    }
    ///set the value and type of an IO entry with an argument
    pub fn set_value_type(&mut self, name: &str, arg: Argument) {
        match self.old_io_names.get(name) {
            Some(IOEntry::In(i)) => {
                self.inputs[*i].value = arg.value;
                self.inputs[*i].ty = arg.ty;
            }
            Some(IOEntry::Out(i)) => {
                self.outputs[*i].value = arg.value;
                self.outputs[*i].ty = arg.ty;
            }
            Some(IOEntry::Node(i)) => {
                self.node_out[*i].value = arg.value;
                self.node_out[*i].ty = arg.ty;
            }
            None => panic!("No entry for {}", name),
        }
    }

    /// Set the type of an IO entry
    pub(crate) fn set_type(&mut self, name: &str, ty: ArgType) {
        match self.old_io_names.get(name) {
            Some(IOEntry::In(i)) => {
                self.inputs[*i].ty = ty;
            }
            Some(IOEntry::Out(i)) => {
                self.outputs[*i].ty = ty;
            }
            Some(IOEntry::Node(i)) => {
                self.node_out[*i].ty = ty;
            }
            None => panic!("No entry for {}", name),
        }
    }

    pub fn get_value(&self, name: &str) -> Option<&Data> {
        match self.old_io_names.get(name) {
            Some(IOEntry::In(i)) => self.inputs[*i].value.as_ref(),
            Some(IOEntry::Out(i)) => self.outputs[*i].value.as_ref(),
            Some(IOEntry::Node(i)) => self.node_out[*i].value.as_ref(),
            None => panic!("No entry for {}", name),
        }
    }

    pub fn set_value(&mut self, name: &str, value: Option<Data>) {
        match self.old_io_names.get(name) {
            Some(IOEntry::In(i)) => {
                self.inputs[*i].value = value;
            }
            //should I restrict this to only graph inputs and node outputs?
            Some(IOEntry::Out(i)) => {
                self.outputs[*i].value = value;
            }
            Some(IOEntry::Node(i)) => {
                self.node_out[*i].value = value;
            }
            None => panic!("No entry for {}", name),
        }
    }

    /// Get the updated name of a Node Input, which should be
    /// either a graph input or a node output.
    /// Will return None if the it isn't a graph input or node output(like an initializer)
    /// Will panic if it's a graph output
    fn mark_input_passed(&mut self, old_name: &str) {
        match self.old_io_names.get(old_name) {
            Some(IOEntry::In(i)) => {
                //FIXME: technically in the spec, initializers are default values
                //for optional inputs, but implementing that would require reworking
                //the way the graph is built, and it's not clear burn users are using initializers
                //in that way
                // see https://github.com/onnx/onnx/issues/2660
                if self.initializers.contains_key(old_name) {
                    log::warn!("triggered by {}\nInitializers as default values is not currently supported", old_name);
                } else {
                    //set the input as passed since a node is referencing it
                    self.inputs[*i].passed = true;
                }
            }
            Some(IOEntry::Out(_)) => {
                panic!(
                    "The input you tried to mark as passed {} is a graph output, check your model",
                    old_name
                )
            }
            Some(IOEntry::Node(i)) => {
                self.node_out[*i].passed = true;
            }
            None => {
                panic!(
                    "Nonexistent entry {:?} cannot be marked as passed",
                    old_name
                );
            }
        }
    }
    pub fn get_arg(&self, name: &str) -> &Argument {
        match self.old_io_names.get(name) {
            Some(IOEntry::In(i)) => self.inputs.get(*i).unwrap(),
            Some(IOEntry::Out(i)) => self.outputs.get(*i).unwrap(),
            Some(IOEntry::Node(i)) => self.node_out.get(*i).unwrap(),
            None => {
                panic!("No entry for {}", name);
            }
        }
    }
}

#[derive(Default)]
pub(crate) struct OnnxGraphBuilder {
    nodes: Vec<Node>,
    io: OnnxGraphIO,
    /// Counter for node names, used for renaming nodes
    node_name_counter: HashMap<NodeType, usize>,
    /// Nodes to remove
    nodes_to_remove: HashSet<usize>,
    /// Map from constant node output names to indices of constant nodes
    constants_map: HashMap<String, usize>,
    constants_types: HashSet<NodeType>,
    /// Map from identity node output names to indices of identity nodes
    identity_idx: HashMap<String, usize>,
}

impl OnnxGraphBuilder {
    pub(crate) fn node_gen(&mut self, model_proto: &ModelProto) {
        self.constants_types = LIFT_CONSTANTS_FOR_NODE_TYPES.into_iter().collect();

        let mut graph_io = OnnxGraphIO::new(
            &model_proto.graph.input,
            &model_proto.graph.output,
            &model_proto.graph.initializer,
        );

        self.nodes = Vec::with_capacity(model_proto.graph.node.len());
        let mut and_idx = 0;
        let mut node_iter = model_proto.graph.node.iter().peekable();

        while let Some(node_proto) = node_iter.next() {
            let mut node = convert_node_proto(node_proto, &mut graph_io);

            remap_node_type(&mut node);

            coalesce(&mut node, &mut node_iter, &mut graph_io);
            self.handle_node_renaming(&mut node);
            self.handle_identity(&mut node, and_idx, &mut graph_io);
            self.check_constants(&mut node, and_idx, &mut graph_io);
            self.handle_unsqueeze(&mut node, &mut graph_io);

            dim_inference(&mut node, &mut graph_io);

            rename_io(&mut node, &mut graph_io);

            self.nodes.push(node);
            and_idx += 1;
        }

        let mut i = 0;
        self.nodes.retain(|_x| {
            let res = !self.nodes_to_remove.contains(&i);
            i += 1;
            res
        });

        // Remove the graph inputs/output that are not used by any node
        self.io = graph_io; //remove_unused_graph_inputs(&graph_io);
    }

    fn handle_node_renaming(&mut self, node: &mut Node) {
        log::debug!("renaming node {:?}", &node.name);
        self.node_name_counter
            .entry(node.node_type.clone())
            .and_modify(|e| *e += 1)
            .or_insert(1);
        let new_name = format!(
            "{}{}",
            node.node_type, self.node_name_counter[&node.node_type]
        )
        .to_lowercase();
        node.name.clone_from(&new_name);
    }

    fn check_constants(&mut self, node: &mut Node, i: usize, graph_io: &mut OnnxGraphIO) {
        if node.node_type == NodeType::Constant
            || (node.node_type == NodeType::Identity
                && graph_io.get_value(&node.inputs[0]).is_some())
        {
            self.constants_map.insert(node.outputs[0].clone(), i);
        } else if self.constants_types.contains(&node.node_type) {
            log::debug!("checking node {} for constants", &node.name);
            for input in node.inputs.iter_mut().skip(1) {
                log::debug!("checking input {:?} for const", input);
                if let Some(const_idx) = self.constants_map.get(input) {
                    let constant = &self.nodes[*const_idx];
                    log::debug!("input {} matched constant node {}", &input, &constant.name);
                    if !constant.inputs.is_empty()
                        && graph_io.get_value(&constant.inputs[0]).is_some()
                    {
                        // The value comes from Identity inputs
                        graph_io.copy_value_type(&constant.inputs[0], input)
                    } else {
                        let arg = convert_constant_value(&constant.attrs);
                        graph_io.set_value_type(input, arg);
                    }
                    self.nodes_to_remove.insert(*const_idx);
                }
            }
        }
    }

    /// Check if the unsqueeze node has a rhs value (rhs is constant) and if not remap it to a reshape
    /// Needs to be called after node renaming to ensure that the rhs name is correct
    /// Needs to be called after constant lifting to ensure that the rhs value exists
    fn handle_unsqueeze(&mut self, node: &mut Node, graph_io: &mut OnnxGraphIO) {
        if node.node_type == NodeType::Unsqueeze
            && node.inputs.len() > 1
            && graph_io.get_value(&node.inputs[1]).is_none()
        {
            remap_unsqueeze_to_reshape(node, graph_io);
        }
    }

    fn handle_identity(&mut self, node: &mut Node, i: usize, graph_io: &OnnxGraphIO) {
        if node.node_type == NodeType::Identity && graph_io.get_value(&node.inputs[0]).is_none() {
            log::debug!("\nfound identity node:\n{:?}\n", &node);
            //map the output name to check for pass through values
            self.identity_idx.insert(node.outputs[0].clone(), i);
            self.nodes_to_remove.insert(i);
        }
    }
}

/// Open an onnx file and convert it to a Graph (intermediate representation)
///
/// # Arguments
///
/// * `onnx_path` - Path to the onnx file
///
/// # Returns
///
/// * `OnnxGraph` - The graph representation of the onnx file
///
/// # Panics
///
/// * If the file cannot be opened
/// * If the file cannot be parsed
/// * If the nodes are not topologically sorted
pub fn parse_onnx(onnx_path: &Path) -> OnnxGraph {
    log::info!("Parsing ONNX file: {}", onnx_path.display());

    // Open the file
    let mut file = File::open(onnx_path).expect("Unable to open file");
    let onnx_model: ModelProto =
        Message::parse_from_reader(&mut file).expect("Unable to parse ONNX file");

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
    let mut builder = OnnxGraphBuilder::default();
    builder.node_gen(&onnx_model);

    let OnnxGraphBuilder {
        nodes,
        io: graph_io,
        ..
    } = builder;

    log::info!("Finished parsing ONNX file: {}", onnx_path.display());

    OnnxGraph { nodes, graph_io }
}

/// Remap the unsqueeze node to a reshape node, Should only be called after
/// node renaming has been done. avoids marking rhs as passed so that it can be
/// properly deleted if nothing else uses it
fn remap_unsqueeze_to_reshape(node: &mut Node, graph_io: &mut OnnxGraphIO) {
    //let out_arg = graph_io.get_node_output(&node.outputs[0]).unwrap();
    match graph_io.get_type(&node.outputs[0]) {
        //TODO: verify deletions. Was taking two references to the same value (output)
        //and then overwriting the second one with the first. Unnecessary work at best
        // and misleading at worst
        ArgType::Tensor(output_tensor) => {
            let inner = output_tensor
                .shape
                .clone()
                .unwrap()
                .into_iter()
                .map(|x| x as i64)
                .collect::<Vec<i64>>();
            let shape_len = inner.len();
            let new_rhs_value = Some(Data::Int64s(inner));
            //moving the remap to here
            let rhs_arg = Argument {
                name: format!("{}_generated_const", node.name),
                ty: ArgType::Tensor(TensorType {
                    elem_type: super::ir::ElementType::Int64,
                    dim: 1,
                    shape: Some(vec![shape_len]),
                }),
                value: new_rhs_value,
                passed: false,
            };
            // ? should this replace the old input (reuse the old key) or should it be a new key
            // going with new key for now
            let rhs_name = rhs_arg.name.clone();
            graph_io.add_generated_const(&rhs_name, rhs_arg);
            node.inputs[1] = rhs_name;
            //node.outputs[0] = out_arg.clone();
            node.node_type = NodeType::Reshape;
        }
        _ => {}
    }
}

/// Rename the inputs and output in the graph and return a map of
/// the old names to the new names.
///
/// The inputs are renamed to be unique and to be in the format of
/// conv2_in1, conv2_in2, etc. This is done to be consistent with
/// the naming convention of the nodes and allow to be used as rust identifiers.
/// Rename the inputs and output in the graph and return a map of
/// the old names to the new names.
fn rename_io(node: &mut Node, graph_io: &mut OnnxGraphIO) {
    log::debug!("checking inputs for node {:?}", &node.name);
    for node_input in node.inputs.iter() {
        graph_io.mark_input_passed(node_input);
    }
    let mut out_count = 1;
    for output in node.outputs.iter_mut() {
        //setting output to passed here since it originally happened at the end of dim inference
        graph_io.set_passed(output);
        let new_name = format!("{}_out{}", node.name, out_count);
        graph_io.update_output_name(output, &new_name);
        out_count += 1;
    }
}

/// Removes the graph inputs/output that are not used by any node.
///
/// In older ONNX models, the inputs and outputs are not always used by the nodes.
/// For example, the input could be used as a state instead of an input. Since the
/// inputs with initializers are moved to the states vector, the inputs vector could
/// contain unused inputs. The same is true for the outputs.
///
/// Generally, it's a good idea to remove unused inputs/outputs because it makes the
/// generated code cleaner and easier to read.
fn remove_unused_graph_inputs(graph_io: &OnnxGraphIO) -> (Vec<Argument>, Vec<Argument>) {
    //NOTE: A better solution would probably be to just filter at generation time
    // as graph_io is only used to generate the burn (or other) graph.
    // This is a quick fix until I can figure out the best way to handle this
    let inputs = graph_io
        .inputs
        .iter()
        .filter(|x| x.passed)
        .cloned()
        .collect::<Vec<_>>();
    let outputs = graph_io
        .outputs
        .iter()
        .filter(|x| x.passed)
        .cloned()
        .collect::<Vec<_>>();
    (inputs, outputs)
}

// Define a trait for topological sorting
trait TopologicalSortable {
    fn is_top_sorted(&self) -> bool;
}

impl TopologicalSortable for Vec<NodeProto> {
    fn is_top_sorted(&self) -> bool {
        // Create a hashmap to store the position of each node in the vector
        let position: HashMap<String, usize> = self
            .iter()
            .enumerate()
            .map(|(idx, node)| (node.name.clone(), idx))
            .collect();

        // Iterate over each node in the vector
        for node in self {
            // Iterate over each output of the node
            for output in &node.output {
                // Iterate over each other node in the vector
                for other_node in self {
                    // If the other node has an input that matches the current output
                    if other_node.input.contains(output) {
                        // If the position of the current node is greater than the position of the other node
                        if position[&node.name] > position[&other_node.name] {
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

/// Get the value of a constant node from its attributes
pub(crate) fn convert_constant_value(attributes: &Attributes) -> Argument {
    // A value can be stored in any of these attributes
    let keys = [
        "value",
        "value_float",
        "value_floats",
        "value_int",
        "value_ints",
        "value_string",
        "value_strings",
        "sparse_value",
    ];

    let value = keys
        .iter()
        .find_map(|&key| attributes.get(key).cloned())
        .expect("Constant should have a value");

    Argument::from(value)
}
