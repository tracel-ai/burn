use std::{
    collections::{HashMap, HashSet},
    fs::File,
    path::Path,
};

use crate::onnx::{
    ir::TensorType, node_remap::remap_node_type, proto_conversion::convert_node_proto,
};

use super::{
    coalesce::coalesce,
    ir::OnnxGraph,
    protos::{ModelProto, TensorProto, ValueInfoProto},
};

use super::dim_inference::dim_inference;
use super::ir::{ArgType, Argument, Node, NodeType, Tensor};

use protobuf::Message;

const LIFT_CONSTANTS_FOR_NODE_TYPES: [NodeType; 7] = [
    NodeType::BatchNormalization,
    NodeType::Clip,
    NodeType::Conv1d,
    NodeType::Conv2d,
    NodeType::Dropout,
    NodeType::Reshape,
    NodeType::Unsqueeze,
];

#[derive(Debug)]
pub(crate) enum IOEntry {
    In(usize),
    Out(usize),
    Node(usize),
}

pub(crate) struct OnnxGraphIO {
    ///Per Onnx spec "Inputs represent graph inputs or values computed elsewhere in the graph..."
    /// Thus all computed inputs are in the list of inputs in a valid Onnx file
    pub(crate) inputs: Vec<Argument>,
    pub(crate) outputs: Vec<Argument>,
    /// Initializers or constants, for more information on why these are treated the
    /// same, please see: https://github.com/onnx/onnx/issues/4677
    pub(crate) constants: HashMap<String, Argument>,
    //pub(crate) initializers: Vec<Argument>,
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
        let inputs = inputs
            .iter()
            .enumerate()
            .map(|(i, x)| {
                let in_name = format!("input{}", in_count);
                old_io_names.insert(x.name.clone(), IOEntry::In(i));
                let mut arg = Argument::try_from(x.clone()).unwrap();
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

        let constants = initializers
            .iter()
            .map(|x| (x.name.clone(), Argument::from_initializer(x)))
            .collect::<HashMap<String, Argument>>();

        Self {
            inputs,
            outputs,
            constants,
            node_out: Vec::new(),
            old_io_names,
        }
    }

    fn update_name(&mut self, arg: &Argument, new_name: &str) {
        match self.old_io_names.get(&arg.name) {
            Some(IOEntry::In(_)) => {
                // let arg = self.inputs.get_mut(*i).unwrap();
                // arg.name = new_name.to_string();
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
                //Constants, Casts
                let idx = self.node_out.len();
                self.old_io_names
                    .insert(arg.name.clone(), IOEntry::Node(idx));
                self.node_out.push(arg.clone());
                self.node_out[idx].name = new_name.to_string();
            }
        }
    }
    fn update_value(&mut self, updated_arg: &Argument) {
        match self.old_io_names.get(&updated_arg.name) {
            Some(IOEntry::Node(i)) => {
                let arg = self.node_out.get_mut(*i).unwrap();
                arg.copy_all_but_name(updated_arg);
            }
            _ => panic!(
                "Tried to update the value of {:?} which was the output from another node",
                &updated_arg.name
            ),
        }
    }

    pub fn init_in(&self, proto_str: &str) -> Argument {
        match self.old_io_names.get(proto_str) {
            None => {
                if let Some(init_arg) = self.constants.get(proto_str) {
                    init_arg.clone()
                } else {
                    Argument::new(proto_str.to_string())
                }
            }

            Some(IOEntry::In(i)) => {
                let mut arg = self.inputs[*i].clone();
                arg.name = proto_str.to_string();
                arg
            }
            Some(IOEntry::Node(i)) => {
                let mut arg = self.node_out[*i].clone();
                arg.name = proto_str.to_string();
                arg
            }
            Some(IOEntry::Out(_)) => {
                panic!("graph out {} can't be an input", &proto_str)
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
    ///Copy data from the graph inputs to the nodes inputs
    pub(crate) fn copy_to_node_inputs(&self, node: &mut Node) {
        for input in node.inputs.iter_mut() {
            if input.name.is_empty() {
                continue;
            }
            match self.old_io_names.get(&input.name) {
                Some(IOEntry::In(i)) => {
                    let arg = self.inputs.get(*i).unwrap();
                    input.copy_all_but_name(arg);
                }
                Some(IOEntry::Out(_i)) => {
                    panic!("Output should only contain final outputs");
                }
                Some(IOEntry::Node(i)) => {
                    let arg = self.node_out.get(*i).unwrap();
                    input.copy_all_but_name(arg);
                }
                None => {
                    //happens with initializers
                    // println!("io names: {:?}", &self.old_io_names);

                    // panic!("Failure when copying nonexistent io to input {} for node {}\nShouldn't happen", &input.name, &node.name);
                }
            }
        }
    }
    ///iterate over the nodes output and copy them to the graph IO
    pub(crate) fn update_tensor_output(&mut self, node: &Node) {
        for node_output in node.outputs.iter() {
            match self.old_io_names.get(&node_output.name) {
                Some(IOEntry::In(i)) => {
                    let arg = self.inputs.get_mut(*i).unwrap();
                    arg.copy_all_but_name(node_output);
                }
                Some(IOEntry::Out(i)) => {
                    let arg = self.outputs.get_mut(*i).unwrap();
                    arg.copy_all_but_name(node_output);
                }
                Some(IOEntry::Node(_)) => {
                    panic!("This output is from another node");
                }
                None => {
                    println!("inserting with name {:?}", &node_output.name);
                    let idx = self.node_out.len();
                    self.old_io_names
                        .insert(node_output.name.clone(), IOEntry::Node(idx));
                    self.node_out.push(node_output.clone());
                }
            }
        }
    }
    // fn add_input(&mut self, old_name: &str, node_idx: usize) {
    //     self.input_of
    //         .entry(old_name.to_string())
    //         .and_modify(|f| f.push(node_idx))
    //         .or_insert(vec![node_idx]);
    // }

    pub(crate) fn get(&self, old_name: &str) -> Option<&Argument> {
        match self.old_io_names.get(old_name) {
            Some(IOEntry::In(i)) => self.inputs.get(*i),
            Some(IOEntry::Out(i)) => self.outputs.get(*i),
            Some(IOEntry::Node(_)) => panic!("This is a node output"),
            None => None,
        }
    }

    fn get_new_name(&self, old_name: &str) -> Option<String> {
        let new_name = match self.old_io_names.get(old_name) {
            Some(IOEntry::In(i)) => Some(self.inputs[*i].name.clone()),
            Some(IOEntry::Out(i)) => Some(self.outputs[*i].name.clone()),
            Some(IOEntry::Node(i)) => Some(self.node_out[*i].name.clone()),
            None => None,
        };
        println!("old name {:?}", &old_name);
        println!("new name value {:?}", &new_name);
        if Some(old_name.to_string()) == new_name {
            println!("old name hasn't changed: {}", old_name);
            None
        } else {
            new_name
        }
    }

    // fn get_node_indices(&self, old_input_name: &str) -> Option<&Vec<usize>> {
    //     self.input_of.get(old_input_name)
    // }
}

#[derive(Default)]
pub(crate) struct ONNXGraphBuilder {
    nodes: Vec<Node>,
    inputs: Vec<Argument>,
    outputs: Vec<Argument>,

    node_name_counter: HashMap<NodeType, usize>,
    //nodes to remove
    nodes_to_remove: HashSet<usize>,
    constants_map: HashMap<String, usize>,

    constants_types: HashSet<NodeType>,
    ///map from old node name to indices of identity nodes
    identity_idx: HashMap<String, usize>,
}

impl ONNXGraphBuilder {
    pub(crate) fn node_gen(&mut self, model_proto: &ModelProto) {
        self.constants_types = LIFT_CONSTANTS_FOR_NODE_TYPES.into_iter().collect();
        // Convert initializers to hashmap for faster lookup
        // let initializers = model_proto
        //     .graph
        //     .initializer
        //     .iter()
        //     .map(|x| (x.name.clone(), x.clone()))
        //     .collect::<HashMap<String, TensorProto>>();

        let mut graph_io = OnnxGraphIO::new(
            &model_proto.graph.input,
            &model_proto.graph.output,
            &model_proto.graph.initializer,
        );

        self.nodes = Vec::with_capacity(model_proto.graph.node.len());
        let mut and_idx = 0;
        let mut node_iter = model_proto.graph.node.iter().peekable();

        while let Some(node_proto) = node_iter.next() {
            let mut node = convert_node_proto(node_proto, &graph_io);

            remap_node_type(&mut node);

            coalesce(&mut node, &mut node_iter, &graph_io);
            self.handle_node_renaming(&mut node);

            self.handle_unsqueeze(&mut node, &graph_io);

            self.handle_identity(&mut node, and_idx);
            self.check_constants(&mut node, and_idx, &mut graph_io);

            if node.node_type != NodeType::Identity {
                dim_inference(&mut node, &mut graph_io);
            }

            //self.handle_coalesce(&mut node, &mut node_iter, and_idx);
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
        let OnnxGraphIO {
            inputs, outputs, ..
        } = graph_io;
        self.inputs = inputs;
        self.outputs = outputs;
    }

    fn handle_node_renaming(&mut self, node: &mut Node) {
        if &node.node_type == &NodeType::Linear {
            println!("rename linear node {:?}", node);
        }
        self.node_name_counter
            .entry(node.node_type.clone())
            .and_modify(|e| *e += 1)
            .or_insert(1);
        let new_name = format!(
            "{}{}",
            node.node_type, self.node_name_counter[&node.node_type]
        )
        .to_lowercase();
        node.name = new_name.clone();
    }

    fn check_constants(&mut self, node: &mut Node, i: usize, graph_io: &mut OnnxGraphIO) {
        if &node.node_type == &NodeType::Constant
            || (&node.node_type == &NodeType::Identity && node.inputs[0].value.is_some())
        {
            self.constants_map.insert(node.outputs[0].name.clone(), i);
        } else if self.constants_types.contains(&node.node_type) {
            println!("lift const type match for node {:?}\n\n", &node);
            for input in node.inputs.iter_mut().skip(1) {
                println!("checking input {:?} for const", input);
                println!("constants map {:?}", &self.constants_map);
                if let Some(const_idx) = self.constants_map.get(&input.name) {
                    println!("\nMATCH\n");
                    let constant = &self.nodes[*const_idx];
                    println!("constant node {:?}", constant);
                    if !constant.inputs.is_empty() && constant.inputs[0].value.is_some() {
                        // The value comes from Identity inputs
                        input.value = constant.inputs[0].value.clone();
                        input.ty = constant.inputs[0].ty.clone();
                        graph_io.update_value(input);
                    } else {
                        let arg = convert_constant_value(constant);
                        input.value = arg.value;
                        input.ty = arg.ty;
                        graph_io.update_value(input);
                    }
                    self.nodes_to_remove.insert(*const_idx);
                    println! {"\nupdated input {:?}", input};
                }
                //TODO: Future me, right now if the constant is written to the
            }
        }
    }

    fn handle_unsqueeze(&mut self, node: &mut Node, graph_io: &OnnxGraphIO) {
        if node.node_type == NodeType::Unsqueeze {
            if let Some(in_arg) = graph_io.get(&node.outputs[0].name) {
                move_output_shape(node, in_arg);
            }
        }
    }

    fn handle_identity(&mut self, node: &mut Node, i: usize) {
        if &node.node_type == &NodeType::Identity && node.inputs[0].value.is_none() {
            println!("\nfound identity node:\n{:?}\n", &node);
            self.identity_idx.insert(node.outputs[0].name.clone(), i);
            self.nodes_to_remove.insert(i);
        } else {
            node.inputs.iter_mut().for_each(|x| {
                if let Some(identity_idx) = self.identity_idx.get(&x.name) {
                    let input_name = &self.nodes[*identity_idx].inputs[0].name;

                    x.name = input_name.clone();
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

    log::debug!("Number of nodes: {:?}", onnx_model.graph.node.len());
    log::debug!("Number of inputs: {:?}", onnx_model.graph.input.len());

    log::debug!(
        "Number of initializers: {:?}",
        onnx_model.graph.initializer.len()
    );

    log::debug!("Number of outputs: {:?}", onnx_model.graph.output.len());
    let mut builder = ONNXGraphBuilder::default();
    builder.node_gen(&onnx_model);

    let ONNXGraphBuilder {
        nodes,
        inputs: mut inner_inputs,
        outputs: mut inner_outputs,
        ..
    } = builder;

    // ONNX nodes must be topologically sorted per spec:
    // https://github.com/onnx/onnx/blob/main/docs/IR.md#graphs
    assert!(nodes.is_top_sorted(), "Nodes are not topologically sorted");

    // Remove the graph inputs/output that are not used by any node
    remove_unused_graph_inputs(&mut inner_inputs, &mut inner_outputs, &nodes);

    log::info!("Finished parsing ONNX file: {}", onnx_path.display());

    OnnxGraph {
        nodes,
        inputs: inner_inputs,
        outputs: inner_outputs,
    }
}

pub(crate) fn move_initializer_data(initializer: &TensorProto, input: &mut Argument) {
    // If the input name matches the tensor name in the initializer
    // Convert the initializer to a tensor
    let tensor = Tensor::try_from(initializer.clone()).expect("Invalid tensor");

    if tensor.dim == 0 {
        // Convert zero dim tensor to scalar
        if let Some(data) = tensor.data {
            input.value = Some(data.into_scalar());
        } else {
            input.value = None;
        }

        // Update the input type
        input.ty = ArgType::Scalar(tensor.elem_type);
    } else {
        // Move the tensor data to the input value
        input.value = tensor.data.clone();

        // Update the input type
        input.ty = ArgType::Tensor(TensorType {
            dim: tensor.dim,
            elem_type: tensor.elem_type,
            shape: tensor.shape,
        });
    }
}

fn move_output_shape(node: &mut Node, out_arg: &Argument) {
    match node.outputs[0].ty {
        ArgType::Tensor(ref mut tensor_type) => {
            if let ArgType::Tensor(arg_tensor) = &out_arg.ty {
                tensor_type.shape = arg_tensor.shape.clone();
            }
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
    println!("checking inputs for node {:?}", &node.name);
    for node_input in node.inputs.iter_mut() {
        println!("old output names {:?}", &graph_io.old_io_names);
        //graph_io.add_input(&node_input.name, i);
        if let Some(input_name) = graph_io.get_new_name(&node_input.name) {
            node_input.passed = true;
            node_input.name = input_name.clone();
        } else {
            node_input.name = "".to_string();
            node_input.passed = false;
        }
    }
    println!("\n\nchecking outputs");
    let mut out_count = 1;
    if node.node_type == NodeType::Constant || node.node_type == NodeType::Identity {
        println!("it's a constant");
        let new_name = format!("{}_out{}", node.name, out_count);
        graph_io.insert(&node.outputs[0], &new_name);
        node.outputs[0].name = new_name;
    } else {
        for output in node.outputs.iter_mut() {
            println!("output name: {}", &output.name);

            let new_name = format!("{}_out{}", node.name, out_count);

            graph_io.update_name(output, &new_name);

            // self.node_output_names
            //     .insert(output.name.clone(), new_name.clone());

            output.name = new_name.clone();
            out_count += 1;
        }
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
fn remove_unused_graph_inputs(
    inputs: &mut Vec<Argument>,
    outputs: &mut Vec<Argument>,
    nodes: &Vec<Node>,
) {
    // Remove inputs that are not used by any node
    inputs.retain(|input| {
        for node in nodes.iter() {
            if node
                .inputs
                .iter()
                .any(|x| x.name == input.name && x.value.is_none())
            {
                return true;
            }
        }
        false
    });

    // Remove outputs that are not used by any node
    outputs.retain(|output| {
        for node in nodes.iter() {
            if node.outputs.iter().any(|x| x.name == output.name) {
                return true;
            }
        }
        false
    });
}

// Define a trait for topological sorting
trait TopologicalSortable {
    fn is_top_sorted(&self) -> bool;
}

impl TopologicalSortable for Vec<Node> {
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
            for output in &node.outputs {
                // Iterate over each other node in the vector
                for other_node in self {
                    // If the other node has an input that matches the current output
                    if other_node.inputs.contains(output) {
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
