use std::{
    env,
    fs::{self, create_dir_all},
    path::{Path, PathBuf},
};

use burn::{
    nn::PReluConfig,
    record::{FullPrecisionSettings, HalfPrecisionSettings, PrecisionSettings},
    tensor::{DataSerialize, Element},
};

use crate::{
    burn::{
        graph::BurnGraph,
        node::{
            avg_pool1d::AvgPool1dNode,
            avg_pool2d::AvgPool2dNode,
            batch_norm::BatchNormNode,
            binary::BinaryNode,
            clip::ClipNode,
            concat::ConcatNode,
            constant::{ConstantNode, ConstantValue, TensorValue},
            conv1d::Conv1dNode,
            conv2d::Conv2dNode,
            conv_transpose_2d::ConvTranspose2dNode,
            dropout::DropoutNode,
            gather::GatherNode,
            global_avg_pool::GlobalAvgPoolNode,
            layer_norm::LayerNormNode,
            linear::LinearNode,
            mask_where::WhereNode,
            matmul::MatmulNode,
            max_pool1d::MaxPool1dNode,
            max_pool2d::MaxPool2dNode,
            prelu::PReluNode,
            reshape::ReshapeNode,
            squeeze::SqueezeNode,
            unary::UnaryNode,
            unsqueeze::UnsqueezeNode,
        },
        ScalarKind, ScalarType, TensorKind, TensorType, Type,
    },
    format_tokens,
    logger::init_log,
    onnx::{
        from_onnx::convert_constant_value,
        ir::{Node, NodeType},
        op_configuration::*,
    },
};

use super::{
    from_onnx::{parse_onnx, OnnxGraphIO},
    ir::{self, ArgType, Argument, Data, ElementType, OnnxGraph},
    op_configuration::{
        avg_pool2d_config, clip_config, concat_config, dropout_config, reshape_config,
        softmax_config,
    },
};

pub use crate::burn::graph::RecordType;

/// Generate code and states from `.onnx` files and save them to the `out_dir`.
#[derive(Debug, Default)]
pub struct ModelGen {
    out_dir: Option<PathBuf>,
    /// List of onnx files to generate source code from.
    inputs: Vec<PathBuf>,
    development: bool,
    half_precision: bool,
    record_type: RecordType,
    embed_states: bool,
}

impl ModelGen {
    /// Create a new `ModelGen`.
    pub fn new() -> Self {
        init_log().ok(); // Error when init multiple times are ignored.
        Self::default()
    }

    /// Set output directory.
    pub fn out_dir(&mut self, out_dir: &str) -> &mut Self {
        self.out_dir = Some(Path::new(out_dir).into());
        self
    }

    /// Add input file.
    pub fn input(&mut self, input: &str) -> &mut Self {
        self.inputs.push(input.into());
        self
    }

    /// Set development mode.
    ///
    /// If this is set to true, the generated model will be saved as `.graph.txt` files and model
    /// states will be saved as `.json` file.
    pub fn development(&mut self, development: bool) -> &mut Self {
        self.development = development;
        self
    }

    /// Run code generation.
    ///
    /// This function is intended to be called from `build.rs` script.
    pub fn run_from_script(&self) {
        self.run(true);
    }

    /// Run code generation.
    ///
    /// This function is intended to be called from CLI.
    pub fn run_from_cli(&self) {
        self.run(false);
    }

    /// Specify parameter precision to be saved.
    ///
    /// # Arguments
    ///
    /// * `half_precision` - If true, half precision is saved. Otherwise, full precision is saved.
    pub fn half_precision(&mut self, half_precision: bool) -> &mut Self {
        self.half_precision = half_precision;
        self
    }

    /// Specify the type of the record to be saved.
    ///
    /// # Arguments
    ///
    /// * `record_type` - The type of the record to be saved.
    pub fn record_type(&mut self, record_type: RecordType) -> &mut Self {
        self.record_type = record_type;
        self
    }

    /// Specify whether to embed states in the generated code.
    ///
    /// # Arguments
    ///
    /// * `embed_states` - If true, states are embedded in the generated code. Otherwise, states are
    /// saved as a separate file.
    pub fn embed_states(&mut self, embed_states: bool) -> &mut Self {
        self.embed_states = embed_states;
        self
    }

    /// Run code generation.
    fn run(&self, is_build_script: bool) {
        log::info!("Starting to convert ONNX to Burn");

        // prepend the out_dir to the cargo_out_dir if this is a build script
        let out_dir = if is_build_script {
            let cargo_out_dir = env::var("OUT_DIR").expect("OUT_DIR env is not set");
            let mut path = PathBuf::from(cargo_out_dir);

            // // Append the out_dir to the cargo_out_dir
            path.push(self.out_dir.clone().unwrap());
            path
        } else {
            self.out_dir.as_ref().expect("out_dir is not set").clone()
        };

        log::debug!("Output directory: {:?}", out_dir);

        create_dir_all(&out_dir).unwrap();

        for input in self.inputs.iter() {
            let file_name = input.file_stem().unwrap();
            let out_file: PathBuf = out_dir.join(file_name);

            log::info!("Converting {:?}", input);
            log::debug!("Input file name: {:?}", file_name);
            log::debug!("Output file: {:?}", out_file);

            self.generate_model(input, out_file);
        }

        log::info!("Finished converting ONNX to Burn");
    }

    /// Generate model source code and model state.
    fn generate_model(&self, input: &PathBuf, out_file: PathBuf) {
        log::info!("Generating model from {:?}", input);
        log::debug!("Development mode: {:?}", self.development);
        log::debug!("Output file: {:?}", out_file);

        let graph = parse_onnx(input.as_ref());

        if self.development {
            // export the graph
            let debug_graph = format!("{:#?}", graph);
            let graph_file = out_file.with_extension("graph.txt");
            log::debug!("Writing debug graph file: {:?}", graph_file);
            fs::write(graph_file, debug_graph).unwrap();
        }

        let blank_space = true;
        let top_comment = Some(format!("Generated from ONNX {input:?} by burn-import"));

        let code = if self.half_precision {
            graph
                .into_burn::<HalfPrecisionSettings>()
                .with_record(out_file.clone(), self.record_type, self.embed_states)
                .with_blank_space(blank_space)
                .with_top_comment(top_comment)
                .codegen()
        } else {
            graph
                .into_burn::<FullPrecisionSettings>()
                .with_record(out_file.clone(), self.record_type, self.embed_states)
                .with_blank_space(blank_space)
                .with_top_comment(top_comment)
                .codegen()
        };

        let code_str = format_tokens(code);
        fs::write(out_file.with_extension("rs"), code_str).unwrap();

        log::info!("Model generated");
    }
}

impl OnnxGraph {
    /// Converts ONNX graph to Burn graph.
    pub fn into_burn<PS: PrecisionSettings + 'static>(self) -> BurnGraph<PS> {
        let mut burn_graph = BurnGraph::<PS>::default();

        let mut unsupported_ops = vec![];

        for node in self.nodes {
            match node.node_type {
                NodeType::Add => burn_graph.register(Self::add_conversion(node, &self.graph_io)),
                NodeType::Sub => burn_graph.register(Self::sub_conversion(node, &self.graph_io)),
                NodeType::Mul => burn_graph.register(Self::mul_conversion(node, &self.graph_io)),
                NodeType::Div => burn_graph.register(Self::div_conversion(node, &self.graph_io)),
                NodeType::Equal => {
                    burn_graph.register(Self::equal_conversion(node, &self.graph_io))
                }
                NodeType::Erf => burn_graph.register(Self::erf_conversion(node, &self.graph_io)),
                NodeType::Exp => burn_graph.register(Self::exp_conversion(node, &self.graph_io)),
                NodeType::Clip => burn_graph.register(Self::clip_conversion(node, &self.graph_io)),
                NodeType::Cos => burn_graph.register(Self::cos_conversion(node, &self.graph_io)),
                NodeType::Conv1d => {
                    burn_graph.register(Self::conv1d_conversion::<PS>(node, &self.graph_io))
                }
                NodeType::Conv2d => {
                    burn_graph.register(Self::conv2d_conversion::<PS>(node, &self.graph_io))
                }
                NodeType::MaxPool1d => {
                    burn_graph.register(Self::max_pool1d_conversion(node, &self.graph_io))
                }
                NodeType::MaxPool2d => {
                    burn_graph.register(Self::max_pool2d_conversion(node, &self.graph_io))
                }
                NodeType::PRelu => {
                    burn_graph.register(Self::prelu_conversion::<PS>(node, &self.graph_io))
                }
                NodeType::AveragePool1d => {
                    burn_graph.register(Self::avg_pool_1d_conversion(node, &self.graph_io))
                }
                NodeType::AveragePool2d => {
                    burn_graph.register(Self::avg_pool_2d_conversion(node, &self.graph_io))
                }
                NodeType::MatMul => {
                    burn_graph.register(Self::matmul_conversion(node, &self.graph_io))
                }
                NodeType::Neg => burn_graph.register(Self::neg_conversion(node, &self.graph_io)),
                NodeType::Not => burn_graph.register(Self::not_conversion(node, &self.graph_io)),
                NodeType::LayerNormalization => {
                    burn_graph.register(Self::layer_norm_conversion::<PS>(node, &self.graph_io))
                }
                NodeType::Linear => {
                    burn_graph.register(Self::linear_conversion::<PS>(node, &self.graph_io))
                }
                NodeType::BatchNormalization => {
                    burn_graph.register(Self::batch_norm_conversion::<PS>(node, &self.graph_io))
                }
                NodeType::Relu => burn_graph.register(Self::relu_conversion(node, &self.graph_io)),
                NodeType::Gelu => burn_graph.register(Self::gelu_conversion(node, &self.graph_io)),
                NodeType::Flatten => {
                    burn_graph.register(Self::flatten_conversion(node, &self.graph_io))
                }
                NodeType::GatherElements => {
                    burn_graph.register(Self::gather_conversion(node, &self.graph_io))
                }
                NodeType::Log => burn_graph.register(Self::log_conversion(node, &self.graph_io)),
                NodeType::LeakyRelu => {
                    burn_graph.register(Self::leaky_relu_conversion(node, &self.graph_io))
                }
                NodeType::LogSoftmax => {
                    burn_graph.register(Self::log_softmax_conversion(node, &self.graph_io))
                }
                NodeType::Softmax => {
                    burn_graph.register(Self::softmax_conversion(node, &self.graph_io))
                }
                NodeType::Sqrt => burn_graph.register(Self::sqrt_conversion(node, &self.graph_io)),
                NodeType::Tanh => burn_graph.register(Self::tanh_conversion(node, &self.graph_io)),
                NodeType::Constant => {
                    burn_graph.register(Self::constant_conversion::<PS>(node, &self.graph_io))
                }
                NodeType::ReduceMax => {
                    burn_graph.register(Self::reduce_max_conversion(node, &self.graph_io))
                }
                NodeType::ReduceMean => {
                    burn_graph.register(Self::reduce_mean_conversion(node, &self.graph_io))
                }
                NodeType::ReduceSum => {
                    burn_graph.register(Self::reduce_sum_conversion(node, &self.graph_io))
                }
                NodeType::Reshape => {
                    burn_graph.register(Self::reshape_conversion(node, &self.graph_io))
                }
                NodeType::Reciprocal => {
                    burn_graph.register(Self::reciprocal_conversion(node, &self.graph_io))
                }
                NodeType::Shape => {
                    burn_graph.register(Self::shape_conversion(node, &self.graph_io))
                }
                NodeType::Sigmoid => {
                    burn_graph.register(Self::sigmoid_conversion(node, &self.graph_io))
                }
                NodeType::Sin => burn_graph.register(Self::sin_conversion(node, &self.graph_io)),
                NodeType::Transpose => {
                    burn_graph.register(Self::transpose_conversion(node, &self.graph_io))
                }
                NodeType::Concat => {
                    burn_graph.register(Self::concat_conversion(node, &self.graph_io))
                }
                NodeType::Cast => burn_graph.register(Self::cast_conversion(node, &self.graph_io)),
                NodeType::Dropout => {
                    burn_graph.register(Self::dropout_conversion(node, &self.graph_io))
                }
                NodeType::GlobalAveragePool => {
                    burn_graph.register(Self::global_avg_pool_conversion(node, &self.graph_io))
                }
                NodeType::ConvTranspose2d => {
                    burn_graph.register(Self::conv_transpose2d_conversion(node, &self.graph_io))
                }
                NodeType::Pow => burn_graph.register(Self::pow_conversion(node, &self.graph_io)),
                NodeType::Unsqueeze => {
                    burn_graph.register(Self::unsqueeze_conversion(node, &self.graph_io))
                }
                NodeType::Where => {
                    burn_graph.register(Self::where_conversion(node, &self.graph_io))
                }
                NodeType::Sign => burn_graph.register(Self::sign_conversion(node, &self.graph_io)),
                NodeType::Squeeze => {
                    burn_graph.register(Self::squeeze_conversion(node, &self.graph_io))
                }
                node_type => unsupported_ops.push(node_type),
            }
        }

        if !unsupported_ops.is_empty() {
            panic!("Unsupported ops: {:?}", unsupported_ops);
        }

        //Get input and output names
        let input_names = self
            .graph_io
            .inputs
            .iter()
            .filter_map(|input| {
                if input.passed {
                    Some(input.name.clone())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let output_names = self
            .graph_io
            .outputs
            .iter()
            .filter_map(|output| {
                if output.passed {
                    Some(output.name.clone())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        // Register inputs and outputs with the graph
        burn_graph.register_input_output(input_names, output_names);

        burn_graph
    }

    fn constant_conversion<PS: PrecisionSettings>(
        node: Node,
        graph_io: &OnnxGraphIO,
    ) -> ConstantNode<PS> {
        let output = node.outputs.first().unwrap();

        let attr = convert_constant_value(&node.attrs);

        let const_value = match attr.ty {
            ArgType::Tensor(tensor) => {
                // Treat tensor with dim 0 as scalar
                if tensor.dim == 0 {
                    panic!("Constant tensor with dim 0 should have been converted to scalar.")
                } else {
                    let kind: TensorKind = tensor.elem_type.clone().into();
                    let dim = tensor.dim;
                    let name = node.name.clone();
                    let shape = tensor.shape.clone();

                    let tensor_value = match tensor.elem_type {
                        // TODO Review how double precision should be supported
                        ElementType::Float32 | ElementType::Float64 => {
                            TensorValue::Float(serialize_data::<PS::FloatElem>(
                                attr.value.unwrap(),
                                tensor.shape.unwrap(),
                            ))
                        }
                        ElementType::Int32 | ElementType::Int64 => {
                            TensorValue::Int(serialize_data::<PS::IntElem>(
                                attr.value.unwrap(),
                                tensor.shape.unwrap(),
                            ))
                        }
                        // TODO support Bool tensor when it is supported by Burn
                        _ => panic!("Unsupported constant tensor type: {:?} ", tensor.elem_type),
                    };

                    ConstantValue::Tensor(TensorType::new(name, dim, kind, shape), tensor_value)
                }
            }
            ArgType::Scalar(elem_type) => match elem_type {
                ElementType::Float64 => ConstantValue::Float64(attr.value.unwrap().into_f64()),
                ElementType::Float32 => ConstantValue::Float32(attr.value.unwrap().into_f32()),
                ElementType::Int32 => ConstantValue::Int32(attr.value.unwrap().into_i32()),
                ElementType::Int64 => ConstantValue::Int64(attr.value.unwrap().into_i64()),
                ElementType::Bool => ConstantValue::Bool(attr.value.unwrap().into_bool()),
                _ => panic!("Unsupported constant tensor type: {:?} ", elem_type),
            },
            ArgType::Shape(_) => panic!("Shape is not supported as constant value."),
        };

        ConstantNode::new(
            node.name.clone(),
            const_value,
            graph_io.get_arg(output).to_type(),
        )
    }

    fn add_conversion(node: Node, graph_io: &OnnxGraphIO) -> BinaryNode {
        let lhs = graph_io.get_arg(node.inputs.first().unwrap()).to_type();
        let rhs = graph_io.get_arg(node.inputs.get(1).unwrap()).to_type();
        let output = graph_io.get_arg(node.outputs.first().unwrap()).to_type();

        BinaryNode::add(lhs, rhs, output)
    }

    fn sub_conversion(node: Node, graph_io: &OnnxGraphIO) -> BinaryNode {
        let lhs = graph_io.get_arg(node.inputs.first().unwrap()).to_type();
        let rhs = graph_io.get_arg(node.inputs.get(1).unwrap()).to_type();
        let output = graph_io.get_arg(node.outputs.first().unwrap()).to_type();

        BinaryNode::sub(lhs, rhs, output)
    }

    fn mul_conversion(node: Node, graph_io: &OnnxGraphIO) -> BinaryNode {
        let lhs = graph_io.get_arg(node.inputs.first().unwrap()).to_type();
        let rhs = graph_io.get_arg(node.inputs.get(1).unwrap()).to_type();
        let output = graph_io.get_arg(node.outputs.first().unwrap()).to_type();

        BinaryNode::mul(lhs, rhs, output)
    }

    fn div_conversion(node: Node, graph_io: &OnnxGraphIO) -> BinaryNode {
        let lhs = graph_io.get_arg(node.inputs.first().unwrap()).to_type();
        let rhs = graph_io.get_arg(node.inputs.get(1).unwrap()).to_type();
        let output = graph_io.get_arg(node.outputs.first().unwrap()).to_type();

        BinaryNode::div(lhs, rhs, output)
    }

    fn matmul_conversion(node: Node, graph_io: &OnnxGraphIO) -> MatmulNode {
        let lhs = graph_io
            .get_arg(node.inputs.first().unwrap())
            .to_tensor_type();
        let rhs = graph_io
            .get_arg(node.inputs.get(1).unwrap())
            .to_tensor_type();
        let output = graph_io
            .get_arg(node.outputs.first().unwrap())
            .to_tensor_type();

        MatmulNode::new(lhs, rhs, output)
    }

    fn equal_conversion(node: Node, graph_io: &OnnxGraphIO) -> BinaryNode {
        let lhs = graph_io.get_arg(node.inputs.first().unwrap()).to_type();
        let rhs = graph_io.get_arg(node.inputs.get(1).unwrap()).to_type();
        let output = graph_io.get_arg(node.outputs.first().unwrap()).to_type();

        BinaryNode::equal(lhs, rhs, output)
    }

    fn erf_conversion(node: Node, graph_io: &OnnxGraphIO) -> UnaryNode {
        let input = graph_io.get_arg(node.inputs.first().unwrap()).to_type();
        let output = graph_io.get_arg(node.outputs.first().unwrap()).to_type();

        UnaryNode::erf(input, output)
    }

    fn leaky_relu_conversion(node: Node, graph_io: &OnnxGraphIO) -> UnaryNode {
        let input = graph_io.get_arg(node.inputs.first().unwrap()).to_type();
        let output = graph_io.get_arg(node.outputs.first().unwrap()).to_type();
        let alpha = leaky_relu_config(&node);

        UnaryNode::leaky_relu(input, output, alpha)
    }

    fn relu_conversion(node: Node, graph_io: &OnnxGraphIO) -> UnaryNode {
        let input = graph_io.get_arg(node.inputs.first().unwrap()).to_type();
        let output = graph_io.get_arg(node.outputs.first().unwrap()).to_type();

        UnaryNode::relu(input, output)
    }

    fn gelu_conversion(node: Node, graph_io: &OnnxGraphIO) -> UnaryNode {
        let input = graph_io.get_arg(node.inputs.first().unwrap()).to_type();
        let output = graph_io.get_arg(node.outputs.first().unwrap()).to_type();

        UnaryNode::gelu(input, output)
    }

    fn log_conversion(node: Node, graph_io: &OnnxGraphIO) -> UnaryNode {
        let input = graph_io.get_arg(node.inputs.first().unwrap()).to_type();
        let output = graph_io.get_arg(node.outputs.first().unwrap()).to_type();

        UnaryNode::log(input, output)
    }

    fn flatten_conversion(node: Node, graph_io: &OnnxGraphIO) -> UnaryNode {
        let input = graph_io.get_arg(node.inputs.first().unwrap()).to_type();
        let output = graph_io.get_arg(node.outputs.first().unwrap()).to_type();
        let (start_dim, end_dim) = flatten_config(&node, graph_io);

        UnaryNode::flatten(input, output, start_dim, end_dim)
    }

    fn gather_conversion(node: Node, graph_io: &OnnxGraphIO) -> GatherNode {
        let input = graph_io
            .get_arg(node.inputs.first().unwrap())
            .to_tensor_type();
        let index = graph_io
            .get_arg(node.inputs.get(1).unwrap())
            .to_tensor_type();
        let output = graph_io
            .get_arg(node.outputs.first().unwrap())
            .to_tensor_type();
        let dim = gather_config(&node, graph_io);

        GatherNode::new(input, index, output, dim)
    }

    fn transpose_conversion(node: Node, graph_io: &OnnxGraphIO) -> UnaryNode {
        let input = graph_io.get_arg(node.inputs.first().unwrap()).to_type();
        let output = graph_io.get_arg(node.outputs.first().unwrap()).to_type();
        let perm = transpose_config(&node, graph_io);

        UnaryNode::transpose(input, output, perm)
    }

    fn cast_conversion(node: Node, graph_io: &OnnxGraphIO) -> UnaryNode {
        let input = graph_io.get_arg(node.inputs.first().unwrap()).to_type();
        let output = graph_io.get_arg(node.outputs.first().unwrap()).to_type();

        UnaryNode::cast(input, output)
    }

    fn reshape_conversion(node: Node, graph_io: &OnnxGraphIO) -> ReshapeNode {
        let input = graph_io
            .get_arg(node.inputs.first().unwrap())
            .to_tensor_type();
        let output = graph_io
            .get_arg(node.outputs.first().unwrap())
            .to_tensor_type();
        let shape = reshape_config(&node, graph_io);

        ReshapeNode::new(input, output, shape)
    }

    fn reduce_max_conversion(node: Node, graph_io: &OnnxGraphIO) -> UnaryNode {
        let input = graph_io.get_arg(node.inputs.first().unwrap()).to_type();
        let output = graph_io.get_arg(node.outputs.first().unwrap()).to_type();
        let dim = reduce_max_config(&node, graph_io);

        UnaryNode::reduce_max(input, output, dim)
    }

    fn reduce_mean_conversion(node: Node, graph_io: &OnnxGraphIO) -> UnaryNode {
        let input = graph_io.get_arg(node.inputs.first().unwrap()).to_type();
        let output = graph_io.get_arg(node.outputs.first().unwrap()).to_type();
        let dim = reduce_mean_config(&node, graph_io);

        UnaryNode::reduce_mean(input, output, dim)
    }

    fn reduce_sum_conversion(node: Node, graph_io: &OnnxGraphIO) -> UnaryNode {
        let input = graph_io.get_arg(node.inputs.first().unwrap()).to_type();
        let output = graph_io.get_arg(node.outputs.first().unwrap()).to_type();
        let dim = reduce_sum_config(&node, graph_io);

        UnaryNode::reduce_sum(input, output, dim)
    }

    fn shape_conversion(node: Node, graph_io: &OnnxGraphIO) -> UnaryNode {
        let input = graph_io.get_arg(node.inputs.first().unwrap()).to_type();
        let output = graph_io.get_arg(node.outputs.first().unwrap()).to_type();
        let (start_dim, end_dim) = shape_config(&node, graph_io);

        UnaryNode::shape(input, output, start_dim, end_dim)
    }

    fn unsqueeze_conversion(node: Node, graph_io: &OnnxGraphIO) -> UnsqueezeNode {
        let input = graph_io.get_arg(node.inputs.first().unwrap()).to_type();
        let output = graph_io
            .get_arg(node.outputs.first().unwrap())
            .to_tensor_type();
        let dims = unsqueeze_config(&node, graph_io);

        UnsqueezeNode::new(input, output, dims)
    }

    fn where_conversion(node: Node, graph_io: &OnnxGraphIO) -> WhereNode {
        let condition = graph_io
            .get_arg(node.inputs.first().unwrap())
            .to_tensor_type();
        let x = graph_io
            .get_arg(node.inputs.get(1).unwrap())
            .to_tensor_type();
        let y = graph_io
            .get_arg(node.inputs.get(2).unwrap())
            .to_tensor_type();
        let output = graph_io
            .get_arg(node.outputs.first().unwrap())
            .to_tensor_type();

        WhereNode::new(condition, x, y, output)
    }

    fn clip_conversion(node: Node, graph_io: &OnnxGraphIO) -> ClipNode {
        let input = graph_io
            .get_arg(node.inputs.first().unwrap())
            .to_tensor_type();
        let output = graph_io
            .get_arg(node.outputs.first().unwrap())
            .to_tensor_type();
        let (min, max) = clip_config(&node, graph_io);

        ClipNode::new(input, output, min, max)
    }

    fn sigmoid_conversion(node: Node, graph_io: &OnnxGraphIO) -> UnaryNode {
        let input = graph_io.get_arg(node.inputs.first().unwrap()).to_type();
        let output = graph_io.get_arg(node.outputs.first().unwrap()).to_type();

        UnaryNode::sigmoid(input, output)
    }

    fn sin_conversion(node: Node, graph_io: &OnnxGraphIO) -> UnaryNode {
        let input = graph_io.get_arg(node.inputs.first().unwrap()).to_type();
        let output = graph_io.get_arg(node.outputs.first().unwrap()).to_type();

        UnaryNode::sin(input, output)
    }

    fn reciprocal_conversion(node: Node, graph_io: &OnnxGraphIO) -> UnaryNode {
        let input = graph_io.get_arg(node.inputs.first().unwrap()).to_type();
        let output = graph_io.get_arg(node.outputs.first().unwrap()).to_type();

        UnaryNode::reciprocal(input, output)
    }

    fn log_softmax_conversion(node: Node, graph_io: &OnnxGraphIO) -> UnaryNode {
        let input = graph_io.get_arg(node.inputs.first().unwrap()).to_type();
        let output = graph_io.get_arg(node.outputs.first().unwrap()).to_type();
        let dim = log_softmax_config(&node, graph_io);

        UnaryNode::log_softmax(input, output, dim)
    }

    fn softmax_conversion(node: Node, graph_io: &OnnxGraphIO) -> UnaryNode {
        let input = graph_io.get_arg(node.inputs.first().unwrap()).to_type();
        let output = graph_io.get_arg(node.outputs.first().unwrap()).to_type();
        let dim = softmax_config(&node, graph_io);

        UnaryNode::softmax(input, output, dim)
    }

    fn sqrt_conversion(node: Node, graph_io: &OnnxGraphIO) -> UnaryNode {
        let input = graph_io.get_arg(node.inputs.first().unwrap()).to_type();
        let output = graph_io.get_arg(node.outputs.first().unwrap()).to_type();

        UnaryNode::sqrt(input, output)
    }

    fn tanh_conversion(node: Node, graph_io: &OnnxGraphIO) -> UnaryNode {
        let input = graph_io.get_arg(node.inputs.first().unwrap()).to_type();
        let output = graph_io.get_arg(node.outputs.first().unwrap()).to_type();

        UnaryNode::tanh(input, output)
    }

    fn concat_conversion(node: Node, graph_io: &OnnxGraphIO) -> ConcatNode {
        let inputs = node
            .inputs
            .iter()
            .map(|input| graph_io.get_arg(input).to_tensor_type())
            .collect();

        let output = graph_io
            .get_arg(node.outputs.first().unwrap())
            .to_tensor_type();
        let dim = concat_config(&node, graph_io);

        ConcatNode::new(inputs, output, dim)
    }

    fn linear_conversion<PS: PrecisionSettings>(
        node: Node,
        graph_io: &OnnxGraphIO,
    ) -> LinearNode<PS> {
        let name = &node.name;
        let input = graph_io
            .get_arg(node.inputs.first().unwrap())
            .to_tensor_type();
        let output = graph_io
            .get_arg(node.outputs.first().unwrap())
            .to_tensor_type();
        let config = linear_config(&node, graph_io);

        let weight = extract_data_serialize::<PS::FloatElem>(1, &node, graph_io)
            .expect("Weight is required");

        let bias = extract_data_serialize::<PS::FloatElem>(2, &node, graph_io);

        LinearNode::new(name, input, output, weight, bias, config)
    }

    fn dropout_conversion(node: Node, graph_io: &OnnxGraphIO) -> DropoutNode {
        let name = &node.name;
        let input = graph_io
            .get_arg(node.inputs.first().unwrap())
            .to_tensor_type();
        let output = graph_io
            .get_arg(node.outputs.first().unwrap())
            .to_tensor_type();
        let config = dropout_config(&node, graph_io);

        DropoutNode::new(name, input, output, config)
    }

    fn batch_norm_conversion<PS: PrecisionSettings>(
        node: Node,
        graph_io: &OnnxGraphIO,
    ) -> BatchNormNode<PS> {
        let config = batch_norm_config(&node, graph_io);
        let input = graph_io
            .get_arg(node.inputs.first().unwrap())
            .to_tensor_type();
        let output = graph_io
            .get_arg(node.outputs.first().unwrap())
            .to_tensor_type();
        let dim = input.dim - 2;

        let gamma =
            extract_data_serialize::<PS::FloatElem>(1, &node, graph_io).expect("Gamma is required");
        let beta =
            extract_data_serialize::<PS::FloatElem>(2, &node, graph_io).expect("Beta is required");
        let running_mean = extract_data_serialize::<PS::FloatElem>(3, &node, graph_io)
            .expect("Running mean is required");
        let running_var = extract_data_serialize::<PS::FloatElem>(4, &node, graph_io)
            .expect("Running var is required");

        let name = &node.name;

        BatchNormNode::new(
            dim,
            name,
            input,
            output,
            gamma,
            beta,
            running_mean,
            running_var,
            config,
        )
    }

    fn layer_norm_conversion<PS: PrecisionSettings>(
        node: Node,
        graph_io: &OnnxGraphIO,
    ) -> LayerNormNode<PS> {
        let (config, full_precision) = layer_norm_config(&node, graph_io);
        let input = graph_io
            .get_arg(node.inputs.first().unwrap())
            .to_tensor_type();
        let output = graph_io
            .get_arg(node.outputs.first().unwrap())
            .to_tensor_type();

        // Scale tensor (aka gamma)
        let gamma =
            extract_data_serialize::<PS::FloatElem>(1, &node, graph_io).expect("Gamma is required");
        // Bias (B) optional tensor
        let beta = extract_data_serialize::<PS::FloatElem>(2, &node, graph_io);

        let name = &node.name;

        LayerNormNode::new(name, input, output, gamma, beta, config, full_precision)
    }

    fn conv1d_conversion<PS: PrecisionSettings>(
        node: Node,
        graph_io: &OnnxGraphIO,
    ) -> Conv1dNode<PS> {
        let input = graph_io
            .get_arg(node.inputs.first().unwrap())
            .to_tensor_type();
        let output = graph_io
            .get_arg(node.outputs.first().unwrap())
            .to_tensor_type();
        let config = conv1d_config(&node, graph_io);

        let bias = node.inputs.len() == 3;
        let weight = extract_data_serialize::<PS::FloatElem>(1, &node, graph_io).unwrap();
        let bias = match bias {
            true => extract_data_serialize::<PS::FloatElem>(2, &node, graph_io),
            false => None,
        };

        let name = &node.name;
        Conv1dNode::<PS>::new(name, input, output, weight, bias, config)
    }

    fn conv2d_conversion<PS: PrecisionSettings>(
        node: Node,
        graph_io: &OnnxGraphIO,
    ) -> Conv2dNode<PS> {
        let input = graph_io
            .get_arg(node.inputs.first().unwrap())
            .to_tensor_type();
        let output = graph_io
            .get_arg(node.outputs.first().unwrap())
            .to_tensor_type();
        let config = conv2d_config(&node, graph_io);

        let bias = node.inputs.len() == 3;
        let weight = extract_data_serialize::<PS::FloatElem>(1, &node, graph_io).unwrap();
        let bias = match bias {
            true => extract_data_serialize::<PS::FloatElem>(2, &node, graph_io),
            false => None,
        };

        let name = &node.name;
        Conv2dNode::<PS>::new(name, input, output, weight, bias, config)
    }
    fn max_pool1d_conversion(node: Node, graph_io: &OnnxGraphIO) -> MaxPool1dNode {
        let input = graph_io
            .get_arg(node.inputs.first().unwrap())
            .to_tensor_type();
        let output = graph_io
            .get_arg(node.outputs.first().unwrap())
            .to_tensor_type();
        let config = max_pool1d_config(&node);

        let name = &node.name;
        MaxPool1dNode::new(name, input, output, config)
    }

    fn max_pool2d_conversion(node: Node, graph_io: &OnnxGraphIO) -> MaxPool2dNode {
        let input = graph_io
            .get_arg(node.inputs.first().unwrap())
            .to_tensor_type();
        let output = graph_io
            .get_arg(node.outputs.first().unwrap())
            .to_tensor_type();
        let config = max_pool2d_config(&node);

        let name = &node.name;
        MaxPool2dNode::new(name, input, output, config)
    }

    fn prelu_conversion<PS: PrecisionSettings>(
        node: Node,
        graph_io: &OnnxGraphIO,
    ) -> PReluNode<PS> {
        let input = graph_io
            .get_arg(node.inputs.first().unwrap())
            .to_tensor_type();
        let output = graph_io
            .get_arg(node.outputs.first().unwrap())
            .to_tensor_type();
        let weight = extract_data_serialize::<PS::FloatElem>(1, &node, graph_io).unwrap();
        let config = PReluConfig::new();
        let name = &node.name;
        PReluNode::<PS>::new(name, input, output, weight, config)
    }
    fn conv_transpose2d_conversion<PS: PrecisionSettings>(
        node: Node,
        graph_io: &OnnxGraphIO,
    ) -> ConvTranspose2dNode<PS> {
        let input = graph_io
            .get_arg(node.inputs.first().unwrap())
            .to_tensor_type();
        let output = graph_io
            .get_arg(node.outputs.first().unwrap())
            .to_tensor_type();
        let config = conv_transpose2d_config(&node, graph_io);

        let bias = node.inputs.len() == 3;
        let weight = extract_data_serialize::<PS::FloatElem>(1, &node, graph_io).unwrap();
        let bias = match bias {
            true => extract_data_serialize::<PS::FloatElem>(2, &node, graph_io),
            false => None,
        };

        let name = &node.name;
        ConvTranspose2dNode::<PS>::new(name, input, output, weight, bias, config)
    }
    fn avg_pool_1d_conversion(node: Node, graph_io: &OnnxGraphIO) -> AvgPool1dNode {
        let input = graph_io
            .get_arg(node.inputs.first().unwrap())
            .to_tensor_type();
        let output = graph_io
            .get_arg(node.outputs.first().unwrap())
            .to_tensor_type();
        let config = avg_pool1d_config(&node);

        let name = &node.name;
        AvgPool1dNode::new(name, input, output, config)
    }

    fn avg_pool_2d_conversion(node: Node, graph_io: &OnnxGraphIO) -> AvgPool2dNode {
        let input = graph_io
            .get_arg(node.inputs.first().unwrap())
            .to_tensor_type();
        let output = graph_io
            .get_arg(node.outputs.first().unwrap())
            .to_tensor_type();
        let config = avg_pool2d_config(&node);

        let name = &node.name;
        AvgPool2dNode::new(name, input, output, config)
    }

    fn global_avg_pool_conversion(node: Node, graph_io: &OnnxGraphIO) -> GlobalAvgPoolNode {
        let input = graph_io
            .get_arg(node.inputs.first().unwrap())
            .to_tensor_type();
        let output = graph_io
            .get_arg(node.outputs.first().unwrap())
            .to_tensor_type();

        let name = &node.name;

        GlobalAvgPoolNode::new(name, input, output)
    }

    fn cos_conversion(node: Node, graph_io: &OnnxGraphIO) -> UnaryNode {
        let input = graph_io.get_arg(node.inputs.first().unwrap()).to_type();
        let output = graph_io.get_arg(node.outputs.first().unwrap()).to_type();

        UnaryNode::cos(input, output)
    }

    fn exp_conversion(node: Node, graph_io: &OnnxGraphIO) -> UnaryNode {
        let input = graph_io.get_arg(node.inputs.first().unwrap()).to_type();
        let output = graph_io.get_arg(node.outputs.first().unwrap()).to_type();

        UnaryNode::exp(input, output)
    }

    fn neg_conversion(node: Node, graph_io: &OnnxGraphIO) -> UnaryNode {
        let input = graph_io.get_arg(node.inputs.first().unwrap()).to_type();
        let output = graph_io.get_arg(node.outputs.first().unwrap()).to_type();
        UnaryNode::neg(input, output)
    }

    fn not_conversion(node: Node, graph_io: &OnnxGraphIO) -> UnaryNode {
        let input = graph_io.get_arg(node.inputs.first().unwrap()).to_type();
        let output = graph_io.get_arg(node.outputs.first().unwrap()).to_type();
        UnaryNode::not(input, output)
    }

    fn pow_conversion(node: Node, graph_io: &OnnxGraphIO) -> BinaryNode {
        let lhs = graph_io.get_arg(node.inputs.first().unwrap()).to_type();
        let rhs = graph_io.get_arg(node.inputs.get(1).unwrap()).to_type();
        let output = graph_io.get_arg(node.outputs.first().unwrap()).to_type();
        match &rhs {
            Type::Tensor(x) => match x.kind {
                TensorKind::Int => BinaryNode::powi(lhs, rhs, output),
                TensorKind::Float => BinaryNode::powf(lhs, rhs, output),
                _ => panic!("pow function requires RHS to be int or float type"),
            },
            Type::Scalar(x) => match x.kind {
                ScalarKind::Int32 | ScalarKind::Int64 => BinaryNode::powi(lhs, rhs, output),
                ScalarKind::Float32 | ScalarKind::Float64 => BinaryNode::powf(lhs, rhs, output),
                _ => panic!("pow function requires RHS to be int or float type"),
            },
            _ => panic!("pow function only supports RHS scalar or tensor types"),
        }
    }

    fn sign_conversion(node: Node, graph_io: &OnnxGraphIO) -> UnaryNode {
        let input = graph_io.get_arg(node.inputs.first().unwrap()).to_type();
        let output = graph_io.get_arg(node.outputs.first().unwrap()).to_type();
        UnaryNode::sign(input, output)
    }

    fn squeeze_conversion(node: Node, graph_io: &OnnxGraphIO) -> SqueezeNode {
        let input = graph_io
            .get_arg(node.inputs.first().unwrap())
            .to_tensor_type();
        let output = graph_io
            .get_arg(node.outputs.first().unwrap())
            .to_tensor_type();
        let axes = squeeze_config(&node, graph_io);

        SqueezeNode::new(input, output, axes)
    }
}

/// Extract data from node states and convert it to `DataSerialize`.
///
/// # Arguments
///
/// * `input_index` - The index of the input originally from input.
/// * `node` - The node where value are stored.
#[track_caller]
fn extract_data_serialize<E: Element>(
    input_index: usize,
    node: &Node,
    graph_io: &OnnxGraphIO,
) -> Option<DataSerialize<E>> {
    if node.inputs.is_empty() {
        return None;
    }

    let input_name = node.inputs.get(input_index);
    input_name?;
    let input_name = input_name.unwrap();

    let ty = graph_io.get_type(input_name);
    println!("input_name: {:?}", input_name);
    println!("ty: {:?}", ty);
    match ty {
        ArgType::Tensor(tensor_type) => {
            let value = graph_io
                .get_value(input_name)
                .expect("Value to be provided.")
                .clone();

            Some(serialize_data(
                value.clone(),
                tensor_type.shape.clone().unwrap(),
            ))
        }
        _ => panic!("Unsupported serialization type"),
    }
}

/// Convert data to `DataSerialize`.
fn serialize_data<E: Element>(data: Data, shape: Vec<usize>) -> DataSerialize<E> {
    match data {
        Data::Float16s(val) => DataSerialize::new(val, shape).convert(),
        Data::Float32s(val) => DataSerialize::new(val, shape).convert(),
        Data::Float64s(val) => DataSerialize::new(val, shape).convert(),
        Data::Int32s(val) => DataSerialize::new(val, shape).convert(),
        Data::Int64s(val) => DataSerialize::new(val, shape).convert(),
        // TODO support Bool tensor when it is supported by Burn
        _ => panic!("Unsupported tensor element type"),
    }
}

impl Argument {
    pub fn to_tensor_type(&self) -> TensorType {
        match &self.ty {
            ArgType::Tensor(ir::TensorType {
                elem_type: ElementType::Float16 | ElementType::Float32 | ElementType::Float64,
                dim,
                ..
            }) => TensorType::new_float(self.name.clone(), *dim),
            ArgType::Tensor(ir::TensorType {
                elem_type: ElementType::Int32 | ElementType::Int64,
                dim,
                ..
            }) => TensorType::new_int(self.name.clone(), *dim),
            ArgType::Tensor(ir::TensorType {
                elem_type: ElementType::Bool,
                dim,
                ..
            }) => TensorType::new_bool(self.name.clone(), *dim),
            _ => panic!("Can't transform to tensor."),
        }
    }

    pub fn to_type(&self) -> Type {
        match &self.ty {
            ArgType::Tensor(tensor) => {
                // Treat tensor with dim 0 as scalar
                if tensor.dim == 0 {
                    Type::Scalar(ScalarType::new(
                        self.name.clone(),
                        ScalarKind::from(&tensor.elem_type),
                    ))
                } else {
                    let kind: TensorKind = tensor.elem_type.clone().into();
                    let dim = tensor.dim;
                    let name = self.name.clone();
                    let shape = tensor.shape.clone();
                    Type::Tensor(TensorType::new(name, dim, kind, shape))
                }
            }

            ArgType::Scalar(elem_type) => {
                Type::Scalar(ScalarType::new(self.name.clone(), elem_type.into()))
            }
            ArgType::Shape(_shape) => panic!("Can't transform shape to tensor."),
        }
    }
}

impl From<&ElementType> for ScalarKind {
    fn from(elem_type: &ElementType) -> Self {
        match elem_type {
            ElementType::Float32 => ScalarKind::Float32,
            ElementType::Float64 => ScalarKind::Float64,
            ElementType::Int32 => ScalarKind::Int32,
            ElementType::Int64 => ScalarKind::Int64,
            ElementType::Bool => ScalarKind::Bool,
            ElementType::String => panic!("String tensor unsupported"),
            ElementType::Float16 => panic!("Float16 tensor unsupported"),
        }
    }
}

impl From<ElementType> for TensorKind {
    fn from(elem_type: ElementType) -> Self {
        match elem_type {
            ElementType::Float32 => TensorKind::Float,
            ElementType::Float64 => TensorKind::Float,
            ElementType::Int32 => TensorKind::Int,
            ElementType::Int64 => TensorKind::Int,
            ElementType::Bool => TensorKind::Bool,
            _ => panic!("Unsupported tensor type"),
        }
    }
}
