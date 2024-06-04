use std::{
    collections::{HashMap, HashSet},
    fs::File,
    path::Path,
};

use crate::onnx::{node_remap::remap_node_type, proto_conversion::convert_node_proto};

use super::{
    coalesce::coalesce, ir::{Data, OnnxGraph, TensorType}, proto_conversion::convert_node_proto2, protos::{ModelProto, NodeProto, TensorProto, ValueInfoProto}
};

use super::dim_inference::dim_inference;
use super::ir::{ArgType, Argument, Node, NodeType};

use protobuf::Message;

const LIFT_CONSTANTS_FOR_NODE_TYPES: [NodeType; 10] = [
    NodeType::BatchNormalization,
    NodeType::Clip,
    NodeType::Conv1d,
    NodeType::Conv2d,
    NodeType::Dropout,
    NodeType::Expand,
    NodeType::Reshape,
    NodeType::Unsqueeze,
    NodeType::ReduceSum,
    NodeType::Squeeze,
];

#[derive(Debug)]
pub(crate) enum IOEntry {
    In(usize),
    Out(usize),
    Node(usize),
}

#[derive(Debug, Clone)]
pub(crate) enum IOEntry2 {
    In(usize),
    Node(usize, usize),
}

pub struct GraphData {
    pub(crate) processed_nodes: Vec<Node>,
    pub(crate) inputs: Vec<Argument>,
    pub(crate) outputs: Vec<Argument>,
    pub(crate) initializers: HashMap<String, Argument>,
    pub(crate) input_name_map: HashMap<String, IOEntry2>,
}

impl GraphData {
    pub(crate) fn new(inputs: &Vec<ValueInfoProto>,
        outputs: &Vec<ValueInfoProto>,
        initializers: &Vec<TensorProto>) -> Self {
        let mut input_name_map = HashMap::new();
        let mut in_count = 1;
        
        let constants = initializers
            .iter()
            .map(|x| (x.name.clone(), Argument::from_initializer(x)))
            .collect::<HashMap<String, Argument>>();
        let outputs = outputs
            .iter()
            .map(|x| {
                Argument::try_from(x.clone()).unwrap()
            })
            .collect::<Vec<Argument>>();
        let inputs = inputs
            .iter()
            .enumerate()
            .map(|(i, x)| {
                let in_name = format!("input{}", in_count);
                input_name_map.insert(x.name.clone(), IOEntry2::In(i));
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
        Self {
            inputs,
            outputs,
            initializers: constants,
            processed_nodes: Vec::new(),
            input_name_map,
        }
    }

    pub fn init_in(&self, proto_str: &str) -> Argument {
        match self.input_name_map.get(proto_str) {
            None => {
                //NOTE: if initializers are guaranteed to be unique, (I think they are
                //need to confirm) then we could pop the initializer from the map
                if let Some(init_arg) = self.initializers.get(proto_str) {
                    init_arg.clone()
                } else {
                  panic!("where did input {:?} come from?", proto_str)  
                }
            }
            Some(IOEntry2::In(i)) => self.inputs[*i].clone(),
            Some(IOEntry2::Node(i, j)) => {
                self.processed_nodes[*i].outputs[*j].clone()
            }
        }
    }

    ///This function does two things:
    /// maps the old output names to the output location
    /// and updates the output names
    pub fn add_node(&mut self, mut node: Node) {
        let mut out_count=1;
        for output in node.outputs.iter_mut() {
            self.input_name_map.insert(output.name.clone(), IOEntry2::Node(self.processed_nodes.len(), 0));
            output.name = format!("{}_out{}", node.name, out_count);
            out_count+=1;
        }
        self.processed_nodes.push(node);
        
    }

    fn consume(mut self) -> (Vec<Node>, Vec<Argument>, Vec<Argument>) {
        self.inputs.retain(|x| x.passed);
        let outputs = self.outputs.into_iter().filter_map(|x| {
            match self.input_name_map.get(&x.name) {
                Some(IOEntry2::Node(_, _)) => Some(x),
                _ => None
            }
        }).collect();
        (self.processed_nodes, self.inputs, outputs)
    }
    
}

pub(crate) struct GraphIO {
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



impl GraphIO {
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

    fn update_name(&mut self, arg: &Argument, new_name: &str) {
        match self.old_io_names.get(&arg.name) {
            Some(IOEntry::In(_)) => {
                panic!("input names are set from the beginning");
            }
            Some(IOEntry::Out(i)) => {
                let arg = self.outputs.get_mut(*i).unwrap();
                arg.name = new_name.to_string();
            }
            Some(IOEntry::Node(i)) => {
                let arg = self.node_out.get_mut(*i).unwrap();
                arg.name = new_name.to_string();
            }
            None => {
                //Constants, Casts wound up here before API changes
                panic!(
                    "Tried to update the name of {} to {} but entry doesn't exist in the map",
                    arg.name, new_name
                )
            }
        }
    }

    /// Used to initialize the input arguments for nodes. Names need to remain the same because
    /// currently the old names are the key for accessing the Argument
    pub fn init_in(&self, proto_str: String) -> Argument {
        match self.old_io_names.get(&proto_str) {
            None => {
                if let Some(init_arg) = self.initializers.get(&proto_str) {
                    init_arg.clone()
                } else {
                    Argument::new(proto_str)
                }
            }

            Some(IOEntry::In(i)) => {
                let mut arg = self.inputs[*i].clone();

                arg.name = proto_str;
                arg.passed = true;
                arg
            }
            Some(IOEntry::Node(i)) => {
                let mut arg = self.node_out[*i].clone();
                arg.name = proto_str;
                arg
            }
            Some(IOEntry::Out(_)) => {
                panic!("graph output {} can't be a Node input", &proto_str)
            }
        }
    }

    fn insert(&mut self, arg: &Argument, new_name: &str) {
        if let Some(idx) = self.old_io_names.get(&arg.name) {
            if let IOEntry::Node(idx) = idx {
                if self.node_out[*idx].name == arg.name {
                    self.node_out[*idx].name = new_name.to_string();
                    return;
                }
            } else {
                panic!("arg entry with old name {} is a graph IO", &arg.name);
            }
        }

        let idx = self.node_out.len();
        self.old_io_names
            .insert(arg.name.clone(), IOEntry::Node(idx));
        self.node_out.push(arg.clone());
        self.node_out[idx].name = new_name.to_string();
    }

    /// Copies node outputs to graph IO. Used at the end of dim inference.
    pub(crate) fn update_tensor_output(&mut self, node: &Node) {
        for node_output in node.outputs.iter() {
            match self.old_io_names.get(&node_output.name) {
                Some(IOEntry::In(i)) => {
                    let arg = self.inputs.get_mut(*i).unwrap();
                    arg.copy_value(node_output);
                }
                Some(IOEntry::Out(i)) => {
                    let arg = self.outputs.get_mut(*i).unwrap();
                    arg.copy_value(node_output);
                    //Set the output to passed since it's been altered by a Node
                    arg.passed = true;
                }
                Some(IOEntry::Node(_)) => {
                    panic!("This output is from another node");
                }
                None => {
                    log::debug!("inserting with name {:?}", &node_output.name);
                    let idx = self.node_out.len();
                    self.old_io_names
                        .insert(node_output.name.clone(), IOEntry::Node(idx));
                    self.node_out.push(node_output.clone());
                }
            }
        }
    }

    /// Used by handle unsqeeze to remap the output of a node to a new name
    /// expected match if it exists is either a graph input or graph output
    pub(crate) fn get_node_output(&self, old_name: &str) -> Option<&Argument> {
        match self.old_io_names.get(old_name) {
            Some(IOEntry::In(i)) => self.inputs.get(*i),
            Some(IOEntry::Out(i)) => self.outputs.get(*i),
            Some(IOEntry::Node(_)) => panic!("This is a node output"),
            None => None,
        }
    }

    /// Get the updated name of a Node Input, which should be
    /// either a graph input or a node output.
    /// Will return None if the it isn't a graph input or node output(like an initializer)
    /// Will panic if it's a graph output
    fn get_new_name(&mut self, old_name: &str) -> Option<String> {
        match self.old_io_names.get(old_name) {
            Some(IOEntry::In(i)) => {
                //FIXME: technically in the spec, initializers are default values
                //for optional inputs, but implementing that would require reworking
                //the way the graph is built, and it's not clear burn users are using initializers
                //in that way
                // see https://github.com/onnx/onnx/issues/2660
                if self.initializers.contains_key(old_name) {
                    None
                } else {
                    //set the input as passed since a node is referencing it
                    self.inputs[*i].passed = true;
                    Some(self.inputs[*i].name.clone())
                }
            }
            Some(IOEntry::Out(_)) => {
                panic!(
                    "you just tried to get an updated name on a graph output: {}",
                    old_name
                )
            }
            Some(IOEntry::Node(i)) => Some(self.node_out[*i].name.clone()),
            None => None,
        }
    }
}

#[derive(Default)]
pub(crate) struct OnnxGraphBuilder {
    processed_nodes: Vec<Node>,
    // inputs: Vec<Argument>,
    // outputs: Vec<Argument>,
    // initializers: HashMap<String, Argument>,
    //input_name_map: HashMap<String, IOEntry2>,
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
    pub(crate) fn node_gen(mut self, model_proto: &ModelProto) -> OnnxGraph {
        self.constants_types = LIFT_CONSTANTS_FOR_NODE_TYPES.into_iter().collect();

        // let mut graph_io = OnnxGraphIO::new(
        //     &model_proto.graph.input,
        //     &model_proto.graph.output,
        //     &model_proto.graph.initializer,
        // );

        let mut graph_data = GraphData::new(
            &model_proto.graph.input,
            &model_proto.graph.output,
            &model_proto.graph.initializer,
        );

        self.processed_nodes = Vec::with_capacity(model_proto.graph.node.len());
        let mut and_idx = 0;
        let mut node_iter = model_proto.graph.node.iter().peekable();

        while let Some(node_proto) = node_iter.next() {
            let mut node = convert_node_proto2(node_proto, &graph_data);

            remap_node_type(&mut node);

            coalesce(&mut node, &mut node_iter, &graph_data);
            self.handle_node_renaming(&mut node);
            // NOTE: start of stateful filter functions
            // args : node, graph_io/data, and_idx
            self.handle_identity(&mut node, and_idx);
            self.check_constants(&mut node, and_idx, &mut graph_data);
            

            self.handle_unsqueeze(&mut node, &graph_data);

            dim_inference(&mut node, &mut graph_data);

            //rename_io(&mut node, &mut graph_io);
            graph_data.add_node(node);
            //self.processed_nodes.push(node);
            and_idx += 1;
        }

        let mut i = 0;
        
        let (mut processed_nodes, inputs, outputs) = graph_data.consume();
        // Remove the graph inputs/output that are not used by any node
        //remove_unused_graph_inputs(&mut inputs, &mut outputs);
        processed_nodes.retain(|_| {let keep = !self.nodes_to_remove.contains(&i); i+=1; keep});
        OnnxGraph {
            nodes: processed_nodes,
            inputs,
            outputs,
        }
    }

    // fn init_graph_io(&mut self, inputs: &Vec<ValueInfoProto>, outputs: &Vec<ValueInfoProto>, initializers: &Vec<TensorProto>) -> GraphData {
    //     let mut in_count = 1;
        
    //     let constants = initializers
    //         .iter()
    //         .map(|x| (x.name.clone(), Argument::from_initializer(x)))
    //         .collect::<HashMap<String, Argument>>();
    //     self.outputs = outputs
    //         .iter()
    //         .map(|x| {
    //             Argument::try_from(x.clone()).unwrap()
    //         })
    //         .collect::<Vec<Argument>>();
    //     self.inputs = inputs
    //         .iter()
    //         .enumerate()
    //         .map(|(i, x)| {
    //             let in_name = format!("input{}", in_count);
    //             self.input_name_map.insert(x.name.clone(), IOEntry2::In(i));
    //             let mut arg = Argument::try_from(x.clone()).unwrap();
    //             if let Some(initial_arg) = constants.get(&x.name) {
    //                 if arg.value.is_none() {
    //                     arg.copy_value(initial_arg);
    //                 }
    //             }

    //             in_count += 1;
    //             arg.name = in_name;
    //             arg
    //         })
    //         .collect::<Vec<Argument>>();
    // }

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

    fn check_constants(&mut self, node: &mut Node, i: usize, _graph_io: &mut GraphData) {
        if node.node_type == NodeType::Constant
            || (node.node_type == NodeType::Identity && node.inputs[0].value.is_some())
        {
            self.constants_map.insert(node.outputs[0].name.clone(), i);
        } else if self.constants_types.contains(&node.node_type) {
            log::debug!("checking node {} for constants", &node.name);
            for input in node.inputs.iter_mut().skip(1) {
                log::debug!("checking input {:?} for const", input);
                if let Some(const_idx) = self.constants_map.get(&input.name) {
                    let constant = &self.processed_nodes[*const_idx];
                    log::debug!(
                        "input {} matched constant node {}",
                        &input.name,
                        &constant.name
                    );
                    if !constant.inputs.is_empty() && constant.inputs[0].value.is_some() {
                        // The value comes from Identity inputs
                        input.value.clone_from(&constant.inputs[0].value);
                        input.ty = constant.inputs[0].ty.clone();
                    } else {
                        let arg = convert_constant_value(constant);
                        input.value = arg.value;
                        input.ty = arg.ty;
                    }
                    self.nodes_to_remove.insert(*const_idx);
                }
            }
        }
    }

    /// Check if the unsqueeze node has a rhs value (rhs is constant) and if not remap it to a reshape
    /// Needs to be called after node renaming to ensure that the rhs name is correct
    /// Needs to be called after constant lifting to ensure that the rhs value exists
    fn handle_unsqueeze(&mut self, node: &mut Node, graph_io: &GraphData) {
        if node.node_type == NodeType::Unsqueeze
            && node.inputs.len() > 1
            && node.inputs[1].value.is_none()
        {
            //if let Some(in_arg) = graph_io.get_node_output(&node.outputs[0].name) {
                remap_unsqueeze_to_reshape(node);
            //}
        }
    }

    fn handle_identity(&mut self, node: &mut Node, i: usize) {
        if node.node_type == NodeType::Identity && node.inputs[0].value.is_none() {
            log::debug!("\nfound identity node:\n{:?}\n", &node);
            //map the output name to check for pass through values
            self.identity_idx.insert(node.outputs[0].name.clone(), i);
            self.nodes_to_remove.insert(i);
        } else {
            //NOTE: it might be possible to rework the API to handle all "per input" operations
            //in a new function that operates on each input.
            node.inputs.iter_mut().for_each(|x| {
                if let Some(identity_idx) = self.identity_idx.get(&x.name) {
                    let input_name = &self.processed_nodes[*identity_idx].inputs[0].name;

                    x.name.clone_from(input_name);
                }
            });
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
    let builder = OnnxGraphBuilder::default();
    let graph = builder.node_gen(&onnx_model);

    // let OnnxGraphBuilder {
    //     processed_nodes: nodes,
    //     inputs: inner_inputs,
    //     outputs: inner_outputs,
    //     ..
    // } = builder;

    log::info!("Finished parsing ONNX file: {}", onnx_path.display());

    // OnnxGraph {
    //     nodes,
    //     inputs: inner_inputs,
    //     outputs: inner_outputs,
    // }
    graph
}


/// Remap the unsqueeze node to a reshape node, Should only be called after
/// node renaming has been done. avoids marking rhs as passed so that it can be
/// properly deleted if nothing else uses it
/// Remap the unsqueeze node to a reshape node
pub(crate) fn remap_unsqueeze_to_reshape(node: &mut Node) {
    match &node.outputs[0].ty {
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
            node.inputs[1] = rhs_arg;
            node.node_type = NodeType::Reshape;
        }
        _ => {}
    }
}
// fn remap_unsqueeze_to_reshape(node: &mut Node) {
//     match node.outputs[0].ty {
//         ArgType::Tensor(ref mut tensor_type) => {
//             if let ArgType::Tensor(arg_tensor) = &out_arg.ty {
//                 tensor_type.shape.clone_from(&arg_tensor.shape);
//                 let inner = arg_tensor
//                     .shape
//                     .clone()
//                     .unwrap()
//                     .into_iter()
//                     .map(|x| x as i64)
//                     .collect::<Vec<i64>>();
//                 let shape_len = inner.len();
//                 let new_rhs_value = Some(Data::Int64s(inner));
//                 //moving the remap to here
//                 let rhs_arg = Argument {
//                     name: format!("{}_generated_const", node.name),
//                     ty: ArgType::Tensor(TensorType {
//                         elem_type: super::ir::ElementType::Int64,
//                         dim: 1,
//                         shape: Some(vec![shape_len]),
//                     }),
//                     value: new_rhs_value,
//                     passed: false,
//                 };
//                 node.inputs[1] = rhs_arg;
//                 node.outputs[0] = out_arg.clone();
//                 node.node_type = NodeType::Reshape;
//             }
//         }
//         _ => {}
//     }
// }

fn set_input_passed(node: &Node, graph_data: &mut GraphData) {
    for input in node.inputs.iter() {
        if let Some(IOEntry2::In(i)) = graph_data.input_name_map.get(&input.name) {
            graph_data.inputs[*i].passed = true;
        }
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
// fn rename_io(node: &mut Node, graph_io: &mut GraphData) {
//     log::debug!("checking inputs for node {:?}", &node.name);
//     for node_input in node.inputs.iter_mut() {
//         if let Some(input_name) = graph_io.get_new_name(&node_input.name) {
//             node_input.passed = true;
//             node_input.name.clone_from(&input_name);
//         } else {
//             node_input.name = "".to_string();
//             node_input.passed = false;
//         }
//     }
//     let mut out_count = 1;
//     if node.node_type == NodeType::Constant || node.node_type == NodeType::Identity {
//         let new_name = format!("{}_out{}", node.name, out_count);
//         graph_io.insert(&node.outputs[0], &new_name);
//         node.outputs[0].name.clone_from(&new_name);
//         log::debug!("Found {} constant", new_name);
//     } else {
//         for output in node.outputs.iter_mut() {
//             log::debug!("output name: {}", &output.name);

//             let new_name = format!("{}_out{}", node.name, out_count);

//             graph_io.update_name(output, &new_name);

//             output.name.clone_from(&new_name);
//             out_count += 1;
//         }
//     }
// }

/// Removes the graph inputs/output that are not used by any node.
///
/// In older ONNX models, the inputs and outputs are not always used by the nodes.
/// For example, the input could be used as a state instead of an input. Since the
/// inputs with initializers are moved to the states vector, the inputs vector could
/// contain unused inputs. The same is true for the outputs.
///
/// Generally, it's a good idea to remove unused inputs/outputs because it makes the
/// generated code cleaner and easier to read.
fn remove_unused_graph_inputs(inputs: &mut Vec<Argument>, outputs: &mut Vec<Argument>) {
    // Remove inputs that are not used by any node
    inputs.retain(|input| input.passed);

    // Remove outputs that are not used by any node
    outputs.retain(|output| output.passed);
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
pub(crate) fn convert_constant_value(node: &Node) -> Argument {
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
        .find_map(|&key| node.attrs.get(key).cloned())
        .expect("Constant should have a value");

    Argument::from(value)
}
