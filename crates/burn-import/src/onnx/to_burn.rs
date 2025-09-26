use std::{
    env,
    fs::{self, create_dir_all},
    path::{Path, PathBuf},
};

use burn::{
    nn::PReluConfig,
    record::{
        DoublePrecisionSettings, FullPrecisionSettings, HalfPrecisionSettings, PrecisionSettings,
    },
    tensor::{Element, TensorData},
};
use log::warn;

use crate::{
    burn::{
        ScalarKind, ScalarType, ShapeType, TensorKind, TensorType, Type,
        graph::BurnGraph,
        node::{
            argmax::ArgMaxNode,
            argmin::ArgMinNode,
            attention::{AttentionNode, AttentionNodeInputs, AttentionNodeOutputs},
            avg_pool1d::AvgPool1dNode,
            avg_pool2d::AvgPool2dNode,
            batch_norm::BatchNormNode,
            bernoulli::BernoulliNode,
            binary::BinaryNode,
            bitshift::{BitShiftNode, Direction},
            bitwiseand::BitwiseAndNode,
            bitwisenot::BitwiseNotNode,
            bitwiseor::BitwiseOrNode,
            bitwisexor::BitwiseXorNode,
            cast::CastNode,
            ceil::CeilNode,
            clip::ClipNode,
            concat::ConcatNode,
            constant::{ConstantNode, ConstantValue},
            constant_of_shape::ConstantOfShapeNode,
            conv_transpose_1d::ConvTranspose1dNode,
            conv_transpose_2d::ConvTranspose2dNode,
            conv_transpose_3d::ConvTranspose3dNode,
            conv1d::Conv1dNode,
            conv2d::Conv2dNode,
            conv3d::Conv3dNode,
            depth_to_space::DepthToSpaceNode,
            dropout::DropoutNode,
            expand::ExpandNode,
            eye_like::EyeLikeNode,
            floor::FloorNode,
            gather::GatherNode,
            gather_elements::GatherElementsNode,
            gemm::GemmNode,
            global_avg_pool::GlobalAvgPoolNode,
            group_norm::GroupNormNode,
            identity::IdentityNode,
            instance_norm::InstanceNormNode,
            layer_norm::LayerNormNode,
            linear::LinearNode,
            matmul::MatmulNode,
            matmul_integer::MatMulIntegerNode,
            max_pool1d::MaxPool1dNode,
            max_pool2d::MaxPool2dNode,
            modulo::ModNode,
            one_hot::OneHotNode,
            pad::PadNode,
            prelu::PReluNode,
            random_normal::RandomNormalNode,
            random_normal_like::RandomNormalLikeNode,
            random_uniform::RandomUniformNode,
            random_uniform_like::RandomUniformLikeNode,
            range::RangeNode,
            reduce::{ReduceNode, ReductionType},
            reshape::ReshapeNode,
            resize::ResizeNode,
            round::RoundNode,
            slice::SliceNode,
            space_to_depth::SpaceToDepthNode,
            split::SplitNode,
            squeeze::SqueezeNode,
            sum::SumNode,
            tile::TileNode,
            top_k::TopKNode,
            trilu::TriluNode,
            unary::UnaryNode,
            unsqueeze::UnsqueezeNode,
            where_op::WhereNode,
        },
    },
    format_tokens,
    logger::init_log,
};

use onnx_ir::{
    convert_constant_value,
    ir::{ArgType, Argument as OnnxArgument, Data, ElementType, Node, NodeType, OnnxGraph},
    node::{
        argmax::argmax_config,
        argmin::argmin_config,
        attention::attention_config,
        avg_pool1d::avg_pool1d_config,
        avg_pool2d::avg_pool2d_config,
        batch_norm::batch_norm_config,
        cast::cast_config,
        clip::clip_config,
        concat::concat_config,
        constant_of_shape::constant_of_shape_config,
        conv_transpose1d::conv_transpose1d_config,
        conv_transpose2d::conv_transpose2d_config,
        conv_transpose3d::conv_transpose3d_config,
        conv1d::conv1d_config,
        conv2d::conv2d_config,
        conv3d::conv3d_config,
        depth_to_space::depth_to_space_config,
        dropout::dropout_config,
        expand::expand_config,
        eye_like::eye_like_config,
        flatten::flatten_config,
        gather::{GatherInput, gather_config},
        gemm::gemm_config,
        group_norm::group_norm_config,
        hard_sigmoid::hard_sigmoid_config,
        instance_norm::instance_norm_config,
        is_inf::is_inf_config,
        layer_norm::layer_norm_config,
        leaky_relu::leaky_relu_config,
        linear::linear_config,
        log_softmax::log_softmax_config,
        max_pool1d::max_pool1d_config,
        max_pool2d::max_pool2d_config,
        modulo::mod_config,
        nonzero::nonzero_config,
        one_hot::one_hot_config,
        pad::pad_config,
        range::range_config,
        reduce::reduce_config,
        reshape::reshape_config,
        resize::resize_config,
        slice::slice_config,
        softmax::softmax_config,
        space_to_depth::space_to_depth_config,
        split::split_config,
        squeeze::squeeze_config,
        tile::tile_config,
        topk::top_k_config,
        transpose::transpose_config,
        trilu::trilu_config,
        unsqueeze::unsqueeze_config,
    },
    parse_onnx,
    util::shape_config,
};

use onnx_ir::node::bitshift::bitshift_config;

pub use crate::burn::graph::RecordType;
use crate::burn::node::mean::MeanNode;
use crate::burn::node::nonzero::NonZeroNode;

/// Generate code and states from `.onnx` files and save them to the `out_dir`.
#[derive(Debug, Default)]
pub struct ModelGen {
    out_dir: Option<PathBuf>,
    /// List of onnx files to generate source code from.
    inputs: Vec<PathBuf>,
    development: bool,
    half_precision: bool,
    double_precision: bool,
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

    /// Set the precision to double floating point precision.
    ///
    /// This uses f64 for floats and i64 for integers, which is necessary for models
    /// with large integer constants that don't fit in i32.
    pub fn double_precision(&mut self, double_precision: bool) -> &mut Self {
        self.double_precision = double_precision;
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
    ///   saved as a separate file.
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

        log::debug!("Output directory: {out_dir:?}");

        create_dir_all(&out_dir).unwrap();

        for input in self.inputs.iter() {
            let file_name = input.file_stem().unwrap();
            let out_file: PathBuf = out_dir.join(file_name);

            log::info!("Converting {input:?}");
            log::debug!("Input file name: {file_name:?}");
            log::debug!("Output file: {out_file:?}");

            self.generate_model(input, out_file);
        }

        log::info!("Finished converting ONNX to Burn");
    }

    /// Generate model source code and model state.
    fn generate_model(&self, input: &PathBuf, out_file: PathBuf) {
        log::info!("Generating model from {input:?}");
        log::debug!("Development mode: {:?}", self.development);
        log::debug!("Output file: {out_file:?}");

        let graph = parse_onnx(input.as_ref());

        if self.development {
            // save onnx graph as a debug file
            let debug_graph = format!("{graph:#?}");
            let graph_file = out_file.with_extension("onnx.txt");
            log::debug!("Writing debug onnx graph file: {graph_file:?}");
            fs::write(graph_file, debug_graph).unwrap();
        }

        let graph = ParsedOnnxGraph(graph);

        if self.development {
            // export the graph
            let debug_graph = format!("{graph:#?}");
            let graph_file = out_file.with_extension("graph.txt");
            log::debug!("Writing debug graph file: {graph_file:?}");
            fs::write(graph_file, debug_graph).unwrap();
        }

        let blank_space = true;
        let top_comment = Some(format!("Generated from ONNX {input:?} by burn-import"));

        let code = if self.double_precision {
            graph
                .into_burn::<DoublePrecisionSettings>()
                .with_record(out_file.clone(), self.record_type, self.embed_states)
                .with_blank_space(blank_space)
                .with_top_comment(top_comment)
                .codegen()
        } else if self.half_precision {
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
        let source_code_file = out_file.with_extension("rs");
        log::info!("Writing source code to {}", source_code_file.display());
        fs::write(source_code_file, code_str).unwrap();

        log::info!("Model generated");
    }
}
#[derive(Debug)]
struct ParsedOnnxGraph(OnnxGraph);
impl ParsedOnnxGraph {
    /// Converts ONNX graph to Burn graph.
    pub fn into_burn<PS: PrecisionSettings + 'static>(self) -> BurnGraph<PS> {
        let mut graph = BurnGraph::<PS>::default();

        let mut unsupported_ops = vec![];

        for node in self.0.nodes {
            match node.node_type {
                NodeType::Add => graph.register(Self::add_conversion(node)),
                NodeType::ArgMax => graph.register(Self::argmax_conversion(node)),
                NodeType::Attention => graph.register(Self::attention_conversion(node)),
                NodeType::BitShift => graph.register(Self::bitshift_conversion(node)),
                NodeType::BitwiseAnd => graph.register(Self::bitwise_and_conversion(node)),
                NodeType::BitwiseOr => graph.register(Self::bitwise_or_conversion(node)),
                NodeType::BitwiseXor => graph.register(Self::bitwise_xor_conversion(node)),
                NodeType::BitwiseNot => graph.register(Self::bitwise_not_conversion(node)),
                NodeType::ArgMin => graph.register(Self::argmin_conversion(node)),
                NodeType::Bernoulli => graph.register(Self::bernoulli_conversion(node)),
                NodeType::Sub => graph.register(Self::sub_conversion(node)),
                NodeType::Mul => graph.register(Self::mul_conversion(node)),
                NodeType::Div => graph.register(Self::div_conversion(node)),
                NodeType::Mod => graph.register(Self::mod_conversion(node)),
                NodeType::Equal => graph.register(Self::equal_conversion(node)),
                NodeType::Erf => graph.register(Self::erf_conversion(node)),
                NodeType::Exp => graph.register(Self::exp_conversion(node)),
                NodeType::Expand => graph.register(Self::expand_conversion(node)),
                NodeType::EyeLike => graph.register(Self::eye_like_conversion(node)),
                NodeType::Floor => graph.register(Self::floor_conversion(node)),
                NodeType::Ceil => graph.register(Self::ceil_conversion(node)),
                NodeType::Clip => graph.register(Self::clip_conversion(node)),
                NodeType::Cos => graph.register(Self::cos_conversion(node)),
                NodeType::Cosh => graph.register(Self::cosh_conversion(node)),
                NodeType::Conv1d => graph.register(Self::conv1d_conversion::<PS>(node)),
                NodeType::Conv2d => graph.register(Self::conv2d_conversion::<PS>(node)),
                NodeType::Conv3d => graph.register(Self::conv3d_conversion::<PS>(node)),
                NodeType::DepthToSpace => graph.register(Self::depth_to_space_conversion(node)),
                NodeType::Max => graph.register(Self::max_conversion(node)),
                NodeType::MaxPool1d => graph.register(Self::max_pool1d_conversion(node)),
                NodeType::MaxPool2d => graph.register(Self::max_pool2d_conversion(node)),
                NodeType::Mean => graph.register(Self::mean_conversion(node)),
                NodeType::PRelu => graph.register(Self::prelu_conversion::<PS>(node)),
                NodeType::AveragePool1d => graph.register(Self::avg_pool_1d_conversion(node)),
                NodeType::AveragePool2d => graph.register(Self::avg_pool_2d_conversion(node)),
                NodeType::MatMul => graph.register(Self::matmul_conversion(node)),
                NodeType::MatMulInteger => graph.register(Self::matmul_integer_conversion(node)),
                NodeType::Neg => graph.register(Self::neg_conversion(node)),
                NodeType::Not => graph.register(Self::not_conversion(node)),
                NodeType::NonZero => graph.register(Self::nonzero_conversion(node)),
                NodeType::And => graph.register(Self::and_conversion(node)),
                NodeType::Or => graph.register(Self::or_conversion(node)),
                NodeType::Xor => graph.register(Self::xor_conversion(node)),
                NodeType::OneHot => graph.register(Self::one_hot_conversion(node)),
                NodeType::Greater => graph.register(Self::greater_conversion(node)),
                NodeType::GreaterOrEqual => graph.register(Self::greater_or_equal_conversion(node)),
                NodeType::Less => graph.register(Self::less_conversion(node)),
                NodeType::LessOrEqual => graph.register(Self::less_or_equal_conversion(node)),
                NodeType::LayerNormalization => {
                    graph.register(Self::layer_norm_conversion::<PS>(node))
                }
                NodeType::InstanceNormalization => {
                    graph.register(Self::instance_norm_conversion::<PS>(node))
                }
                NodeType::Linear => graph.register(Self::linear_conversion::<PS>(node)),
                NodeType::BatchNormalization => {
                    graph.register(Self::batch_norm_conversion::<PS>(node))
                }
                NodeType::GroupNormalization => {
                    graph.register(Self::group_norm_conversion::<PS>(node))
                }
                NodeType::Relu => graph.register(Self::relu_conversion(node)),
                NodeType::Gelu => graph.register(Self::gelu_conversion(node)),
                NodeType::Flatten => graph.register(Self::flatten_conversion(node)),
                NodeType::Gather => graph.register(Self::gather_conversion(node)),
                NodeType::GatherElements => graph.register(Self::gather_elements_conversion(node)),
                NodeType::HardSigmoid => graph.register(Self::hard_sigmoid_conversion(node)),
                NodeType::Log => graph.register(Self::log_conversion(node)),
                NodeType::LeakyRelu => graph.register(Self::leaky_relu_conversion(node)),
                NodeType::LogSoftmax => graph.register(Self::log_softmax_conversion(node)),
                NodeType::Softmax => graph.register(Self::softmax_conversion(node)),
                NodeType::Sqrt => graph.register(Self::sqrt_conversion(node)),
                NodeType::Tan => graph.register(Self::tan_conversion(node)),
                NodeType::Tanh => graph.register(Self::tanh_conversion(node)),
                NodeType::Constant => graph.register(Self::constant_conversion::<PS>(node)),
                NodeType::Min => graph.register(Self::min_conversion(node)),
                NodeType::Range => graph.register(Self::range_conversion(node)),
                NodeType::ReduceMax => graph.register(Self::reduce_max_conversion(node)),
                NodeType::ReduceMin => graph.register(Self::reduce_min_conversion(node)),
                NodeType::ReduceMean => graph.register(Self::reduce_mean_conversion(node)),
                NodeType::ReduceProd => graph.register(Self::reduce_prod_conversion(node)),
                NodeType::ReduceSum => graph.register(Self::reduce_sum_conversion(node)),
                NodeType::ReduceSumSquare => {
                    graph.register(Self::reduce_sum_square_conversion(node))
                }
                NodeType::ReduceL1 => graph.register(Self::reduce_l1_conversion(node)),
                NodeType::ReduceL2 => graph.register(Self::reduce_l2_conversion(node)),
                NodeType::ReduceLogSum => graph.register(Self::reduce_log_sum_conversion(node)),
                NodeType::ReduceLogSumExp => {
                    graph.register(Self::reduce_log_sum_exp_conversion(node))
                }
                NodeType::Reshape => graph.register(Self::reshape_conversion(node)),
                NodeType::Resize => graph.register(Self::resize_conversion(node)),
                NodeType::Reciprocal => graph.register(Self::reciprocal_conversion(node)),
                NodeType::Round => graph.register(Self::round_conversion(node)),
                NodeType::Shape => graph.register(Self::shape_conversion(node)),
                NodeType::Sigmoid => graph.register(Self::sigmoid_conversion(node)),
                NodeType::Sin => graph.register(Self::sin_conversion(node)),
                NodeType::Sinh => graph.register(Self::sinh_conversion(node)),
                NodeType::Size => graph.register(Self::size_conversion(node)),
                NodeType::Slice => graph.register(Self::slice_conversion(node)),
                NodeType::SpaceToDepth => graph.register(Self::space_to_depth_conversion(node)),
                NodeType::Sum => graph.register(Self::sum_conversion(node)),
                NodeType::Transpose => graph.register(Self::transpose_conversion(node)),
                NodeType::Concat => graph.register(Self::concat_conversion(node)),
                NodeType::Cast => graph.register(Self::cast_conversion(node)),
                NodeType::Dropout => graph.register(Self::dropout_conversion(node)),
                NodeType::GlobalAveragePool => {
                    graph.register(Self::global_avg_pool_conversion(node))
                }
                NodeType::ConvTranspose1d => {
                    graph.register(Self::conv_transpose1d_conversion::<PS>(node))
                }
                NodeType::ConvTranspose2d => {
                    graph.register(Self::conv_transpose2d_conversion::<PS>(node))
                }
                NodeType::ConvTranspose3d => {
                    graph.register(Self::conv_transpose3d_conversion::<PS>(node))
                }
                NodeType::Pad => graph.register(Self::pad_conversion(node)),
                NodeType::Pow => graph.register(Self::pow_conversion(node)),
                NodeType::Unsqueeze => graph.register(Self::unsqueeze_conversion(node)),
                NodeType::Where => graph.register(Self::where_conversion(node)),
                NodeType::Sign => graph.register(Self::sign_conversion(node)),
                NodeType::Squeeze => graph.register(Self::squeeze_conversion(node)),
                NodeType::RandomUniform => graph.register(Self::random_uniform_conversion(node)),
                NodeType::RandomUniformLike => {
                    graph.register(Self::random_uniform_like_conversion(node))
                }
                NodeType::Tile => graph.register(Self::tile_conversion(node)),
                NodeType::TopK => graph.register(Self::top_k_conversion(node)),
                NodeType::Trilu => graph.register(Self::trilu_conversion(node)),
                NodeType::RandomNormal => graph.register(Self::random_normal_conversion(node)),
                NodeType::RandomNormalLike => {
                    graph.register(Self::random_normal_like_conversion(node))
                }
                NodeType::ConstantOfShape => {
                    graph.register(Self::constant_of_shape_conversion(node))
                }
                NodeType::Split => graph.register(Self::split_conversion(node)),
                NodeType::Gemm => graph.register(Self::gemm_conversion(node)),
                NodeType::IsNaN => graph.register(Self::is_nan_conversion(node)),
                NodeType::IsInf => graph.register(Self::is_inf_conversion(node)),
                NodeType::Identity => graph.register(Self::identity_conversion(node)),
                NodeType::Abs => graph.register(Self::abs_conversion(node)),
                node_type => unsupported_ops.push(node_type),
            }
        }

        if !unsupported_ops.is_empty() {
            panic!("Unsupported ops: {unsupported_ops:?}");
        }

        // Get input and output names
        let input_names = self
            .0
            .inputs
            .iter()
            .map(|input| input.name.clone())
            .collect::<Vec<_>>();

        let output_names = self
            .0
            .outputs
            .iter()
            .map(|output| output.name.clone())
            .collect::<Vec<_>>();

        // Register inputs and outputs with the graph
        graph.register_input_output(input_names, output_names);

        graph
    }

    fn constant_conversion<PS: PrecisionSettings>(node: Node) -> ConstantNode {
        let output = node.outputs.first().unwrap();
        let attr = convert_constant_value(&node);

        // Helper to map elem type to ConstantValue (single scalar)
        fn scalar_from_data(elem: ElementType, data: onnx_ir::ir::Data) -> ConstantValue {
            match elem {
                ElementType::Float64 => ConstantValue::Float64(data.into_f64()),
                ElementType::Float32 => ConstantValue::Float32(data.into_f32()),
                ElementType::Int64 => ConstantValue::Int64(data.into_i64()),
                ElementType::Int32 => ConstantValue::Int32(data.into_i32()),
                ElementType::Bool => ConstantValue::Bool(data.into_bool()),
                // If you want to allow 8-bit scalars too:
                ElementType::Uint8 => ConstantValue::Int32(data.into_i32()), // or define UInt8 variant if you have one
                ElementType::Int8 => ConstantValue::Int32(data.into_i32()),
                _ => panic!("Unsupported scalar type: {elem:?}"),
            }
        }

        let const_value = match &output.ty {
            // Shape constants already handled
            ArgType::Shape(rank) => {
                let shape_data = attr.value.expect("Shape constant should have value");
                let shape_values: Vec<usize> = shape_data
                    .data
                    .into_i64s()
                    .into_iter()
                    .map(|v| v as usize)
                    .collect();
                assert_eq!(shape_values.len(), *rank, "Shape constant rank mismatch");
                ConstantValue::Shape(shape_values)
            }

            ArgType::Tensor(tensor) => {
                // Accept rank-0 tensor constants as SCALARS instead of panicking.
                if tensor.rank == 0 {
                    let v = attr
                        .value
                        .as_ref()
                        .expect("Scalar constant should have value");
                    scalar_from_data(tensor.elem_type.clone(), v.data.clone())
                } else {
                    let kind: TensorKind = tensor.elem_type.clone().into();
                    let rank = tensor.rank;
                    let name = node.name.clone();
                    let tensor_data = attr.value.expect("Constant tensor should have value");

                    let tensor_data = match &tensor.elem_type {
                        ElementType::Float32 | ElementType::Float64 | ElementType::Float16 => {
                            serialize_data::<PS::FloatElem>(tensor_data.data, tensor_data.shape)
                        }
                        ElementType::Int32
                        | ElementType::Int64
                        | ElementType::Uint8
                        | ElementType::Int8 => {
                            serialize_data::<PS::IntElem>(tensor_data.data, tensor_data.shape)
                        }
                        ElementType::Bool => {
                            // Handle boolean tensor constants
                            serialize_bool_data(tensor_data.data, tensor_data.shape)
                        }
                        other => panic!("Unsupported constant tensor type: {:?} ", other),
                    };

                    ConstantValue::Tensor(TensorType::new(name, rank, kind), tensor_data)
                }
            }

            ArgType::Scalar(elem_type) => {
                // Scalar output already typed as scalar → just map from Data.
                let v = attr.value.unwrap();
                match elem_type {
                    ElementType::Float64 => ConstantValue::Float64(v.data.into_f64()),
                    ElementType::Float32 => ConstantValue::Float32(v.data.into_f32()),
                    ElementType::Int32 => ConstantValue::Int32(v.data.into_i32()),
                    ElementType::Int64 => ConstantValue::Int64(v.data.into_i64()),
                    ElementType::Bool => ConstantValue::Bool(v.data.into_bool()),
                    other => panic!("Unsupported constant scalar type: {other:?} "),
                }
            }
        };

        // IMPORTANT:
        // If you hit a rank-0 tensor but output.ty is still ArgType::Tensor(rank=0),
        // ConstantValue above is a Scalar. ConstantNode::new expects a Type for the output.
        // Ensure Type::from(output) can represent scalars. If it can't, override here:
        let out_ty = match (&output.ty, &const_value) {
            (
                ArgType::Tensor(t),
                ConstantValue::Float32(_)
                | ConstantValue::Float64(_)
                | ConstantValue::Int32(_)
                | ConstantValue::Int64(_)
                | ConstantValue::Bool(_),
            ) if t.rank == 0 => {
                // Convert to scalar Type explicitly
                // (Adjust constructors to your Type/ScalarType API)
                let scalar_kind = match t.elem_type {
                    ElementType::Float32 => {
                        ScalarType::new(output.name.clone(), ScalarKind::Float32)
                    }
                    ElementType::Float64 => {
                        ScalarType::new(output.name.clone(), ScalarKind::Float64)
                    }
                    ElementType::Int32 => ScalarType::new(output.name.clone(), ScalarKind::Int32),
                    ElementType::Int64 => ScalarType::new(output.name.clone(), ScalarKind::Int64),
                    ElementType::Uint8 => ScalarType::new(output.name.clone(), ScalarKind::Int32), // or define UInt8 variant if you have one
                    ElementType::Int8 => ScalarType::new(output.name.clone(), ScalarKind::Int32),
                    ElementType::Bool => ScalarType::new(output.name.clone(), ScalarKind::Bool),
                    _ => panic!("Unsupported scalar type for output: {:?}", t.elem_type),
                };
                Type::Scalar(scalar_kind)
            }
            _ => Type::from(output),
        };

        ConstantNode::new(node.name.clone(), const_value, out_ty)
    }
    fn random_uniform_conversion(node: Node) -> RandomUniformNode {
        let output = node.outputs.first().unwrap();
        let output_type = TensorType::from(output);

        let high = node
            .attrs
            .get("high")
            .map(|val| val.clone().into_f32() as f64)
            .unwrap_or(1.0f64);
        let low = node
            .attrs
            .get("low")
            .map(|val| val.clone().into_f32() as f64)
            .unwrap_or(0.0f64);

        let shape = node
            .attrs
            .get("shape")
            .map(|val| {
                val.clone()
                    .into_i64s()
                    .into_iter()
                    .map(|elem| elem as usize)
                    .collect()
            })
            .expect("Missing required 'shape' attribute");

        if node.attrs.contains_key("seed") {
            warn!("The 'seed' attribute is not supported");
        }

        RandomUniformNode::new(output_type, low, high, shape)
    }

    fn identity_conversion(node: Node) -> IdentityNode {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());

        IdentityNode::new(input, output)
    }

    fn random_uniform_like_conversion(node: Node) -> RandomUniformLikeNode {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());
        let low = node
            .attrs
            .get("low")
            .map(|val| val.clone().into_f32() as f64)
            .unwrap_or(0.0f64); // default is 0.0
        let high = node
            .attrs
            .get("high")
            .map(|val| val.clone().into_f32() as f64)
            .unwrap_or(1.0f64); // default is 1.0

        if node.attrs.contains_key("seed") {
            warn!("seed attribute is not supported!");
        }

        RandomUniformLikeNode::new(low, high, input, output)
    }

    fn random_normal_conversion(node: Node) -> RandomNormalNode {
        let output = node.outputs.first().unwrap();
        let output_type = TensorType::from(output);

        let mean = node
            .attrs
            .get("mean")
            .map(|val| val.clone().into_f32() as f64)
            .unwrap_or(0.0f64);
        let scale = node
            .attrs
            .get("scale")
            .map(|val| val.clone().into_f32() as f64)
            .unwrap_or(1.0f64);

        let shape = node
            .attrs
            .get("shape")
            .map(|val| {
                val.clone()
                    .into_i64s()
                    .into_iter()
                    .map(|elem| elem as usize)
                    .collect()
            })
            .expect("Missing required 'shape' attribute");

        if node.attrs.contains_key("seed") {
            warn!("The 'seed' attribute is not supported");
        }

        RandomNormalNode::new(output_type, mean, scale, shape)
    }

    fn random_normal_like_conversion(node: Node) -> RandomNormalLikeNode {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());
        let mean = node
            .attrs
            .get("mean")
            .map(|val| val.clone().into_f32() as f64)
            .unwrap_or(0.0f64);
        let scale = node
            .attrs
            .get("scale")
            .map(|val| val.clone().into_f32() as f64)
            .unwrap_or(1.0f64);

        if node.attrs.contains_key("seed") {
            warn!("seed attribute is not supported!");
        }

        RandomNormalLikeNode::new(mean, scale, input, output)
    }

    pub(crate) fn constant_of_shape_conversion(node: Node) -> ConstantOfShapeNode {
        // Additional types needed for ConstantOfShape:
        use crate::burn::node::constant_of_shape::ConstantValue;

        // Get the shape configuration from onnx-ir
        let shape = constant_of_shape_config(&node);

        let output = Type::from(node.outputs.first().unwrap());

        // The value of the output elements.Should be a one-element tensor.
        // If not specified, it defaults to a tensor of value 0 and datatype float32
        // https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConstantOfShape
        let value = node
            .attrs
            .get("value")
            .map(|val| val.clone().into_tensor().data)
            .map(|val_data| match val_data {
                // TODO: Handle Float16
                Data::Float32s(vals) => ConstantValue::from_vec(vals),
                Data::Float64s(vals) => ConstantValue::from_vec(vals),
                Data::Int32s(vals) => ConstantValue::from_vec(vals),
                Data::Int64s(vals) => ConstantValue::from_vec(vals),
                Data::Bools(vals) => ConstantValue::from_vec(vals),
                ty => panic!("Unsupported value type {ty:?} for ConstantOfShape!"),
            })
            .unwrap_or(ConstantValue::Float32(0.0f32));

        ConstantOfShapeNode::new(shape, output, value)
    }

    fn add_conversion(node: Node) -> BinaryNode {
        let lhs = Type::from(node.inputs.first().unwrap());
        let rhs = Type::from(node.inputs.get(1).unwrap());
        let output = Type::from(node.outputs.first().unwrap());

        BinaryNode::add(lhs, rhs, output)
    }

    fn sub_conversion(node: Node) -> BinaryNode {
        let lhs = Type::from(node.inputs.first().unwrap());
        let rhs = Type::from(node.inputs.get(1).unwrap());
        let output = Type::from(node.outputs.first().unwrap());

        BinaryNode::sub(lhs, rhs, output)
    }

    fn mul_conversion(node: Node) -> BinaryNode {
        let lhs_arg = node.inputs.first().unwrap();
        let rhs_arg = node.inputs.get(1).unwrap();
        let output_arg = node.outputs.first().unwrap();

        log::debug!(
            "mul_conversion for {}: lhs={:?}, rhs={:?}",
            node.name,
            lhs_arg,
            rhs_arg
        );

        let lhs = Type::from(lhs_arg);
        let rhs = Type::from(rhs_arg);
        let output = Type::from(output_arg);

        BinaryNode::mul(lhs, rhs, output)
    }

    fn div_conversion(node: Node) -> BinaryNode {
        let lhs = Type::from(node.inputs.first().unwrap());
        let rhs = Type::from(node.inputs.get(1).unwrap());
        let output = Type::from(node.outputs.first().unwrap());

        BinaryNode::div(lhs, rhs, output)
    }

    fn mod_conversion(node: Node) -> ModNode {
        let lhs = Type::from(node.inputs.first().unwrap());
        let rhs = Type::from(node.inputs.get(1).unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());
        let config = mod_config(&node);

        ModNode::new(lhs, rhs, output, config.fmod)
    }

    fn matmul_conversion(node: Node) -> MatmulNode {
        let lhs = TensorType::from(node.inputs.first().unwrap());
        let rhs = TensorType::from(node.inputs.get(1).unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());

        MatmulNode::new(lhs, rhs, output)
    }
    fn matmul_integer_conversion(node: Node) -> MatMulIntegerNode {
        use crate::burn::{TensorKind, TensorType};
        use onnx_ir::ir::{ArgType as OnnxArgType, TensorType as OnnxTensorType};

        // Burn-side types for codegen
        let lhs = TensorType::from(node.inputs.first().unwrap()); // u8 or i8
        let rhs = TensorType::from(node.inputs.get(1).unwrap()); // u8 or i8
        let lhs_zp = node.inputs.get(2).map(TensorType::from); // scalar or [K]
        let rhs_zp = node.inputs.get(3).map(TensorType::from); // scalar or [N]

        // Output must be i32
        let mut output = TensorType::from(node.outputs.first().unwrap());
        output.kind = TensorKind::Int;
        // If you track width, set it here (e.g., output.d = 32);

        // ---- Validate zero-point vector lengths using IR shapes (if available) ----
        // Get IR view of A and B to read static_shape
        let a_ir = node.inputs.first().unwrap();
        let b_ir = node.inputs.get(1).unwrap();

        let a_shape = match &a_ir.ty {
            OnnxArgType::Tensor(OnnxTensorType { static_shape, .. }) => static_shape.as_ref(),
            _ => None,
        };
        let b_shape = match &b_ir.ty {
            OnnxArgType::Tensor(OnnxTensorType { static_shape, .. }) => static_shape.as_ref(),
            _ => None,
        };

        // K = last dim of A (when viewed as 2-D), N = first dim of B
        let k_dim = a_shape.and_then(|s| {
            if !s.is_empty() {
                s.last().copied()
            } else {
                None
            }
        });
        let n_dim = b_shape.and_then(|s| {
            if !s.is_empty() {
                s.first().copied()
            } else {
                None
            }
        });

        // Collapse vec_len_if_1d_ir
        fn vec_len_if_1d_ir(arg: &onnx_ir::ir::Argument) -> Option<usize> {
            if let OnnxArgType::Tensor(OnnxTensorType {
                rank, static_shape, ..
            }) = &arg.ty
                && *rank == 1
            {
                return static_shape.as_ref().and_then(|s| s.first().copied());
            }
            None
        }

        // Collapse a_zero_point check
        if let Some(a_zp_ir) = node.inputs.get(2)
            && let Some(zp_len) = vec_len_if_1d_ir(a_zp_ir)
            && let Some(k) = k_dim
        {
            // Zero point can be scalar (length 1, broadcast) or per-channel (length K)
            assert!(
                zp_len == 1 || zp_len == k,
                "MatMulInteger: a_zero_point length {} must be 1 (scalar) or K {} (cols of A)",
                zp_len,
                k
            );
        }
        // Scalars are fine; no check needed.

        // Collapse b_zero_point check
        if let Some(b_zp_ir) = node.inputs.get(3)
            && let Some(zp_len) = vec_len_if_1d_ir(b_zp_ir)
            && let Some(n) = n_dim
        {
            // Zero point can be scalar (length 1, broadcast) or per-channel (length N)
            assert!(
                zp_len == 1 || zp_len == n,
                "MatMulInteger: b_zero_point length {} must be 1 (scalar) or N {} (cols of B)",
                zp_len,
                n
            );
        }
        // Scalars are fine; no check needed.

        MatMulIntegerNode::new(lhs, rhs, lhs_zp, rhs_zp, output)
    }

    fn equal_conversion(node: Node) -> BinaryNode {
        let lhs = Type::from(node.inputs.first().unwrap());
        let rhs = Type::from(node.inputs.get(1).unwrap());
        let output = Type::from(node.outputs.first().unwrap());

        BinaryNode::equal(lhs, rhs, output)
    }

    fn bitshift_conversion(node: Node) -> BitShiftNode {
        let inputs = node.inputs.iter().map(Type::from).collect();
        let output = Type::from(node.outputs.first().unwrap());
        let onnx_direction = bitshift_config(&node);

        // Map ONNX direction to burn-import Direction
        let direction = match onnx_direction {
            onnx_ir::node::bitshift::Direction::Left => Direction::Left,
            onnx_ir::node::bitshift::Direction::Right => Direction::Right,
        };

        BitShiftNode::new(inputs, output, direction)
    }

    fn bitwise_and_conversion(node: Node) -> BitwiseAndNode {
        let inputs = node.inputs.iter().map(Type::from).collect();
        let output = Type::from(node.outputs.first().unwrap());

        BitwiseAndNode::new(inputs, output)
    }

    fn bitwise_or_conversion(node: Node) -> BitwiseOrNode {
        let inputs = node.inputs.iter().map(Type::from).collect();
        let output = Type::from(node.outputs.first().unwrap());

        BitwiseOrNode::new(inputs, output)
    }

    fn bitwise_xor_conversion(node: Node) -> BitwiseXorNode {
        let inputs = node.inputs.iter().map(Type::from).collect();
        let output = Type::from(node.outputs.first().unwrap());

        BitwiseXorNode::new(inputs, output)
    }

    fn bitwise_not_conversion(node: Node) -> BitwiseNotNode {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());

        BitwiseNotNode::new(input, output)
    }

    fn max_conversion(node: Node) -> BinaryNode {
        let lhs = Type::from(node.inputs.first().unwrap());
        let rhs = Type::from(node.inputs.get(1).unwrap());
        let output = Type::from(node.outputs.first().unwrap());

        BinaryNode::max_pair(lhs, rhs, output)
    }

    fn erf_conversion(node: Node) -> UnaryNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());

        UnaryNode::erf(input, output)
    }

    fn leaky_relu_conversion(node: Node) -> UnaryNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());
        let alpha = leaky_relu_config(&node);

        UnaryNode::leaky_relu(input, output, alpha)
    }

    fn hard_sigmoid_conversion(node: Node) -> UnaryNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());
        let (alpha, beta) = hard_sigmoid_config(&node);

        UnaryNode::hard_sigmoid(input, output, alpha, beta)
    }

    fn relu_conversion(node: Node) -> UnaryNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());

        UnaryNode::relu(input, output)
    }

    fn gelu_conversion(node: Node) -> UnaryNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());

        UnaryNode::gelu(input, output)
    }

    fn log_conversion(node: Node) -> UnaryNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());

        UnaryNode::log(input, output)
    }

    fn flatten_conversion(node: Node) -> UnaryNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());
        let axis = flatten_config(&node);

        UnaryNode::flatten(input, output, axis)
    }

    fn gather_conversion(node: Node) -> GatherNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());
        let config = gather_config(&node);

        // Create GatherNode based on whether indices are static or runtime
        match config.indices {
            GatherInput::Static(indices) => {
                GatherNode::with_static_indices(input, indices, output, config.axis)
            }
            GatherInput::Runtime(arg) => {
                let index = Type::from(&arg);
                GatherNode::new(input, index, output, config.axis)
            }
        }
    }

    fn gather_elements_conversion(node: Node) -> GatherElementsNode {
        let input = TensorType::from(node.inputs.first().unwrap());
        let index = TensorType::from(node.inputs.get(1).unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());
        let config = gather_config(&node);

        GatherElementsNode::new(input, index, output, config.axis)
    }

    fn transpose_conversion(node: Node) -> UnaryNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());
        let perm = transpose_config(&node);

        UnaryNode::transpose(input, output, perm)
    }

    fn cast_conversion(node: Node) -> CastNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());
        let config = cast_config(&node);

        CastNode::new(input, output, config.to)
    }

    fn reshape_conversion(node: Node) -> ReshapeNode {
        let input_arg = node.inputs.first().unwrap();
        let output_arg = node.outputs.first().unwrap();
        let output = Type::from(output_arg);
        let config = reshape_config(&node);

        // Convert input to appropriate Type
        let input = Type::from(input_arg);

        match config.shape {
            onnx_ir::node::reshape::ReshapeInput::Static(shape) => {
                ReshapeNode::new(input, output, shape)
            }
            onnx_ir::node::reshape::ReshapeInput::Runtime(shape_arg) => {
                let shape_input = Type::from(&shape_arg);
                ReshapeNode::new(input, output, shape_input)
            }
        }
    }

    fn resize_conversion(node: Node) -> ResizeNode {
        let name = &node.name;

        let input = TensorType::from(&node.inputs[0]);

        let output = TensorType::from(node.outputs.first().unwrap());

        let config = resize_config(&node);

        // Convert from onnx-ir types to burn types
        let mode = match config.mode {
            onnx_ir::node::resize::ResizeMode::Nearest => {
                crate::burn::node::resize::ResizeMode::Nearest
            }
            onnx_ir::node::resize::ResizeMode::Linear => {
                crate::burn::node::resize::ResizeMode::Linear
            }
            onnx_ir::node::resize::ResizeMode::Cubic => {
                crate::burn::node::resize::ResizeMode::Cubic
            }
        };

        let scales = config.scales.map(|s| match s {
            onnx_ir::node::resize::ResizeScales::Static(s) => {
                crate::burn::node::resize::ResizeScales::Static(s)
            }
            onnx_ir::node::resize::ResizeScales::Runtime(arg) => {
                crate::burn::node::resize::ResizeScales::Runtime(Type::from(&arg))
            }
        });

        let sizes = config.sizes.map(|s| match s {
            onnx_ir::node::resize::ResizeSizes::Static(s) => {
                crate::burn::node::resize::ResizeSizes::Static(s)
            }
            onnx_ir::node::resize::ResizeSizes::Runtime(arg) => {
                crate::burn::node::resize::ResizeSizes::Runtime(Type::from(&arg))
            }
        });

        ResizeNode::new(name, input, output, mode, scales, sizes)
    }

    fn min_conversion(node: Node) -> BinaryNode {
        let lhs = Type::from(node.inputs.first().unwrap());
        let rhs = Type::from(node.inputs.get(1).unwrap());
        let output = Type::from(node.outputs.first().unwrap());

        BinaryNode::min_pair(lhs, rhs, output)
    }

    fn range_conversion(node: Node) -> RangeNode {
        use crate::burn::node::range::RangeParam;
        use onnx_ir::node::range::RangeInput;

        let config = range_config(&node);
        let output = TensorType::from(node.outputs.first().unwrap());

        let start = match config.start {
            RangeInput::Static(value) => RangeParam::Static(value),
            RangeInput::Runtime(arg) => RangeParam::Runtime(Type::from(&arg)),
        };

        let limit = match config.limit {
            RangeInput::Static(value) => RangeParam::Static(value),
            RangeInput::Runtime(arg) => RangeParam::Runtime(Type::from(&arg)),
        };

        let delta = match config.delta {
            RangeInput::Static(value) => RangeParam::Static(value),
            RangeInput::Runtime(arg) => RangeParam::Runtime(Type::from(&arg)),
        };

        log::debug!(
            "Range node conversion: start={:?}, limit={:?}, delta={:?}",
            start,
            limit,
            delta
        );

        RangeNode::new(start, limit, delta, output)
    }

    fn reduce_max_conversion(node: Node) -> ReduceNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());
        let config = reduce_config(&node);

        ReduceNode::new(input, output, ReductionType::Max, config)
    }

    fn reduce_min_conversion(node: Node) -> ReduceNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());
        let config = reduce_config(&node);

        ReduceNode::new(input, output, ReductionType::Min, config)
    }

    fn reduce_mean_conversion(node: Node) -> ReduceNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());
        let config = reduce_config(&node);

        ReduceNode::new(input, output, ReductionType::Mean, config)
    }

    fn reduce_prod_conversion(node: Node) -> ReduceNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());
        let config = reduce_config(&node);

        ReduceNode::new(input, output, ReductionType::Prod, config)
    }

    fn reduce_sum_conversion(node: Node) -> ReduceNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());
        let config = reduce_config(&node);

        ReduceNode::new(input, output, ReductionType::Sum, config)
    }

    fn reduce_sum_square_conversion(node: Node) -> ReduceNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());
        let config = reduce_config(&node);

        ReduceNode::new(input, output, ReductionType::SumSquare, config)
    }

    fn reduce_l1_conversion(node: Node) -> ReduceNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());
        let config = reduce_config(&node);

        ReduceNode::new(input, output, ReductionType::L1, config)
    }

    fn reduce_l2_conversion(node: Node) -> ReduceNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());
        let config = reduce_config(&node);

        ReduceNode::new(input, output, ReductionType::L2, config)
    }

    fn reduce_log_sum_conversion(node: Node) -> ReduceNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());
        let config = reduce_config(&node);

        ReduceNode::new(input, output, ReductionType::LogSum, config)
    }

    fn reduce_log_sum_exp_conversion(node: Node) -> ReduceNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());
        let config = reduce_config(&node);

        ReduceNode::new(input, output, ReductionType::LogSumExp, config)
    }

    fn shape_conversion(node: Node) -> UnaryNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());
        let (start_dim, end_dim) = shape_config(&node);

        UnaryNode::shape(input, output, start_dim, end_dim)
    }

    fn unsqueeze_conversion(node: Node) -> UnsqueezeNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());
        let axes = unsqueeze_config(&node);
        UnsqueezeNode::new(input, output, axes)
    }

    fn where_conversion(node: Node) -> WhereNode {
        let condition = Type::from(node.inputs.first().unwrap());
        let x = Type::from(node.inputs.get(1).unwrap());
        let y = Type::from(node.inputs.get(2).unwrap());
        let output = Type::from(node.outputs.first().unwrap());

        WhereNode::new(condition, x, y, output)
    }

    fn clip_conversion(node: Node) -> ClipNode {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());
        let (min, max) = clip_config(&node);

        ClipNode::new(input, output, min, max)
    }

    fn sigmoid_conversion(node: Node) -> UnaryNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());

        UnaryNode::sigmoid(input, output)
    }

    fn sin_conversion(node: Node) -> UnaryNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());

        UnaryNode::sin(input, output)
    }

    fn sinh_conversion(node: Node) -> UnaryNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());

        UnaryNode::sinh(input, output)
    }

    fn size_conversion(node: Node) -> UnaryNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());

        UnaryNode::size(input, output)
    }

    fn slice_conversion(node: Node) -> SliceNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());
        let config = slice_config(&node);

        use crate::burn::node::slice::SliceParam;
        use onnx_ir::node::slice::SliceInput;

        // Convert starts parameter
        let starts_param = match config.starts {
            SliceInput::Static(values) => SliceParam::Static(values),
            SliceInput::Runtime(arg) => SliceParam::Runtime(Type::from(&arg)),
        };

        // Convert ends parameter
        let ends_param = match config.ends {
            SliceInput::Static(values) => SliceParam::Static(values),
            SliceInput::Runtime(arg) => SliceParam::Runtime(Type::from(&arg)),
        };

        let mut slice_node = SliceNode::new(input, output, starts_param, ends_param);

        // Convert axes parameter if present
        if let Some(axes) = config.axes {
            let axes_param = match axes {
                SliceInput::Static(values) => SliceParam::Static(values),
                SliceInput::Runtime(arg) => SliceParam::Runtime(Type::from(&arg)),
            };
            slice_node = slice_node.with_axes(axes_param);
        }

        // Convert steps parameter if present
        if let Some(steps) = config.steps {
            let steps_param = match steps {
                SliceInput::Static(values) => SliceParam::Static(values),
                SliceInput::Runtime(arg) => SliceParam::Runtime(Type::from(&arg)),
            };
            slice_node = slice_node.with_steps(steps_param);
        }

        slice_node
    }

    fn space_to_depth_conversion(node: Node) -> SpaceToDepthNode {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());
        let block_size = space_to_depth_config(&node);

        SpaceToDepthNode::new(input, output, block_size)
    }

    fn sum_conversion(node: Node) -> SumNode {
        let inputs = node.inputs.iter().map(TensorType::from).collect();
        let output = TensorType::from(node.outputs.first().unwrap());

        SumNode::new(inputs, output)
    }

    fn reciprocal_conversion(node: Node) -> UnaryNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());

        UnaryNode::reciprocal(input, output)
    }

    fn log_softmax_conversion(node: Node) -> UnaryNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());
        let dim = log_softmax_config(&node);

        UnaryNode::log_softmax(input, output, dim)
    }

    fn softmax_conversion(node: Node) -> UnaryNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());
        let dim = softmax_config(&node);

        UnaryNode::softmax(input, output, dim)
    }

    fn sqrt_conversion(node: Node) -> UnaryNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());

        UnaryNode::sqrt(input, output)
    }

    fn abs_conversion(node: Node) -> UnaryNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());

        UnaryNode::abs(input, output)
    }

    fn tan_conversion(node: Node) -> UnaryNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());

        UnaryNode::tan(input, output)
    }

    fn tanh_conversion(node: Node) -> UnaryNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());

        UnaryNode::tanh(input, output)
    }

    fn argmax_conversion(node: Node) -> ArgMaxNode {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());
        let config = argmax_config(&node);

        ArgMaxNode::new(input, output, config.axis, config.keepdims)
    }

    fn argmin_conversion(node: Node) -> ArgMinNode {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());
        let config = argmin_config(&node);

        ArgMinNode::new(input, output, config.axis, config.keepdims)
    }

    fn bernoulli_conversion(node: Node) -> BernoulliNode {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());

        BernoulliNode::new(input, output)
    }

    fn concat_conversion(node: Node) -> ConcatNode {
        let inputs: Vec<Type> = node.inputs.iter().map(Type::from).collect();
        let output = Type::from(node.outputs.first().unwrap());
        let dim = concat_config(&node);
        ConcatNode::new(inputs, output, dim)
    }

    fn linear_conversion<PS: PrecisionSettings>(node: Node) -> LinearNode {
        let name = &node.name;
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());
        let config = linear_config(&node);

        let weight = extract_data_serialize::<PS::FloatElem>(1, &node).expect("Weight is required");

        let bias = extract_data_serialize::<PS::FloatElem>(2, &node);

        LinearNode::new(name, input, output, weight, bias, config)
    }

    fn dropout_conversion(node: Node) -> DropoutNode {
        let name = &node.name;
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());
        let config = dropout_config(&node);

        DropoutNode::new(name, input, output, config)
    }

    fn batch_norm_conversion<PS: PrecisionSettings>(node: Node) -> BatchNormNode {
        let config = batch_norm_config(&node);
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());
        let dim = input.rank - 2;

        let gamma = extract_data_serialize::<PS::FloatElem>(1, &node).expect("Gamma is required");
        let beta = extract_data_serialize::<PS::FloatElem>(2, &node).expect("Beta is required");
        let running_mean =
            extract_data_serialize::<PS::FloatElem>(3, &node).expect("Running mean is required");
        let running_var =
            extract_data_serialize::<PS::FloatElem>(4, &node).expect("Running var is required");

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

    fn layer_norm_conversion<PS: PrecisionSettings>(node: Node) -> LayerNormNode {
        let (config, full_precision) = layer_norm_config(&node);
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());

        // Scale tensor (aka gamma)
        let gamma = extract_data_serialize::<PS::FloatElem>(1, &node).expect("Gamma is required");
        // Bias (B) optional tensor
        let beta = extract_data_serialize::<PS::FloatElem>(2, &node);

        let name = &node.name;

        LayerNormNode::new(name, input, output, gamma, beta, config, full_precision)
    }

    fn instance_norm_conversion<PS: PrecisionSettings>(node: Node) -> InstanceNormNode {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());

        // Get configuration from onnx-ir
        let config = instance_norm_config(&node);
        // Scale tensor (aka gamma)
        let gamma = extract_data_serialize::<PS::FloatElem>(1, &node).expect("Gamma is required");
        // Bias (B) optional tensor
        let beta = extract_data_serialize::<PS::FloatElem>(2, &node).expect("Beta is required");

        let name = &node.name;
        InstanceNormNode::new(name, input, output, gamma, beta, config)
    }

    fn group_norm_conversion<PS: PrecisionSettings>(node: Node) -> GroupNormNode {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());

        // Get configuration from onnx-ir
        let (config, full_precision) = group_norm_config(&node);
        // Scale tensor (aka gamma)
        let gamma = extract_data_serialize::<PS::FloatElem>(1, &node).expect("Gamma is required");
        // Bias (B) optional tensor
        let beta = extract_data_serialize::<PS::FloatElem>(2, &node).expect("Beta is required");

        let name = &node.name;
        GroupNormNode::new(name, input, output, gamma, beta, config, full_precision)
    }

    fn conv1d_conversion<PS: PrecisionSettings>(node: Node) -> Conv1dNode {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());

        // Get configuration from onnx-ir
        let config = conv1d_config(&node);

        let bias = node.inputs.len() == 3;
        let weight = extract_data_serialize::<PS::FloatElem>(1, &node).unwrap();
        let bias = match bias {
            true => extract_data_serialize::<PS::FloatElem>(2, &node),
            false => None,
        };

        let name = &node.name;
        Conv1dNode::new(name, input, output, weight, bias, config)
    }

    fn conv2d_conversion<PS: PrecisionSettings>(node: Node) -> Conv2dNode {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());
        let config = conv2d_config(&node);

        let bias = node.inputs.len() == 3;
        let weight = extract_data_serialize::<PS::FloatElem>(1, &node).unwrap();
        let bias = match bias {
            true => extract_data_serialize::<PS::FloatElem>(2, &node),
            false => None,
        };

        let name = &node.name;
        Conv2dNode::new(name, input, output, weight, bias, config)
    }

    fn conv3d_conversion<PS: PrecisionSettings>(node: Node) -> Conv3dNode {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());
        let config = conv3d_config(&node);

        let bias = node.inputs.len() == 3;
        let weight = extract_data_serialize::<PS::FloatElem>(1, &node).unwrap();
        let bias = match bias {
            true => extract_data_serialize::<PS::FloatElem>(2, &node),
            false => None,
        };

        let name = &node.name;
        Conv3dNode::new(name, input, output, weight, bias, config)
    }

    fn depth_to_space_conversion(node: Node) -> DepthToSpaceNode {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());
        let config = depth_to_space_config(&node);

        DepthToSpaceNode::new(input, output, config)
    }

    fn max_pool1d_conversion(node: Node) -> MaxPool1dNode {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());

        // Get configuration from onnx-ir
        let config = max_pool1d_config(&node);

        let name = &node.name;
        MaxPool1dNode::new(name, input, output, config)
    }

    fn max_pool2d_conversion(node: Node) -> MaxPool2dNode {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());
        let config = max_pool2d_config(&node);

        let name = &node.name;
        MaxPool2dNode::new(name, input, output, config)
    }

    fn mean_conversion(node: Node) -> MeanNode {
        let inputs = node.inputs.iter().map(TensorType::from).collect();
        let output = TensorType::from(node.outputs.first().unwrap());

        MeanNode::new(inputs, output)
    }

    fn prelu_conversion<PS: PrecisionSettings>(node: Node) -> PReluNode {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());
        let mut weight =
            extract_data_serialize::<PS::FloatElem>(1, &node).expect("PRelu weight is required");
        let name = &node.name;

        // Determine weight shape and flatten if necessary
        let weight_shape = if weight.shape.len() > 1 {
            let trailing_dims_product: usize = weight.shape[1..].iter().product();

            if trailing_dims_product == 1 {
                // Flatten to rank 1 as Burn expects
                weight.shape = vec![weight.shape[0]];
                weight.shape[0]
            } else {
                panic!(
                    "PRelu weight shape {:?} is invalid. Expected shape [C] or [C, 1, ...] where trailing dimensions are 1",
                    weight.shape
                );
            }
        } else if weight.shape.is_empty() {
            // Scalar weight
            1
        } else {
            // Already rank 1
            weight.shape[0]
        };

        let config = PReluConfig::new().with_num_parameters(weight_shape);

        PReluNode::new(name, input, output, weight, config)
    }

    fn conv_transpose1d_conversion<PS: PrecisionSettings>(node: Node) -> ConvTranspose1dNode {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());

        // Get configuration from onnx-ir
        let onnx_config = conv_transpose1d_config(&node);

        // Convert to burn ConvTranspose1dConfig
        let config = burn::nn::conv::ConvTranspose1dConfig::new(
            [onnx_config.channels_in, onnx_config.channels_out],
            onnx_config.kernel_size,
        )
        .with_stride(onnx_config.stride)
        .with_padding(onnx_config.padding)
        .with_dilation(onnx_config.dilation)
        .with_padding_out(onnx_config.padding_out)
        .with_groups(onnx_config.groups)
        .with_bias(onnx_config.bias);

        let bias = node.inputs.len() == 3;
        let weight = extract_data_serialize::<PS::FloatElem>(1, &node).unwrap();
        let bias = match bias {
            true => extract_data_serialize::<PS::FloatElem>(2, &node),
            false => None,
        };

        let name = &node.name;
        ConvTranspose1dNode::new(name, input, output, weight, bias, config)
    }

    fn conv_transpose2d_conversion<PS: PrecisionSettings>(node: Node) -> ConvTranspose2dNode {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());
        let config = conv_transpose2d_config(&node);

        let bias = node.inputs.len() == 3;
        let weight = extract_data_serialize::<PS::FloatElem>(1, &node).unwrap();
        let bias = match bias {
            true => extract_data_serialize::<PS::FloatElem>(2, &node),
            false => None,
        };

        let name = &node.name;
        ConvTranspose2dNode::new(name, input, output, weight, bias, config)
    }
    fn conv_transpose3d_conversion<PS: PrecisionSettings>(node: Node) -> ConvTranspose3dNode {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());
        let config = conv_transpose3d_config(&node);

        let bias = node.inputs.len() == 3;
        let weight = extract_data_serialize::<PS::FloatElem>(1, &node).unwrap();
        let bias = match bias {
            true => extract_data_serialize::<PS::FloatElem>(2, &node),
            false => None,
        };

        let name = &node.name;
        ConvTranspose3dNode::new(name, input, output, weight, bias, config)
    }
    fn avg_pool_1d_conversion(node: Node) -> AvgPool1dNode {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());

        // Get configuration from onnx-ir
        let config = avg_pool1d_config(&node);

        let name = &node.name;
        AvgPool1dNode::new(name, input, output, config)
    }

    fn avg_pool_2d_conversion(node: Node) -> AvgPool2dNode {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());
        let config = avg_pool2d_config(&node);

        let name = &node.name;
        AvgPool2dNode::new(name, input, output, config)
    }

    fn global_avg_pool_conversion(node: Node) -> GlobalAvgPoolNode {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());

        let name = &node.name;

        GlobalAvgPoolNode::new(name, input, output)
    }

    fn cos_conversion(node: Node) -> UnaryNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());

        UnaryNode::cos(input, output)
    }

    fn cosh_conversion(node: Node) -> UnaryNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());

        UnaryNode::cosh(input, output)
    }

    fn exp_conversion(node: Node) -> UnaryNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());

        UnaryNode::exp(input, output)
    }

    fn expand_conversion(node: Node) -> ExpandNode {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());
        let shape = expand_config(&node);
        ExpandNode::new(input, output, shape)
    }

    fn eye_like_conversion(node: Node) -> EyeLikeNode {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());
        let config = eye_like_config(&node);
        EyeLikeNode::new(input, output, config)
    }

    fn neg_conversion(node: Node) -> UnaryNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());
        UnaryNode::neg(input, output)
    }

    fn not_conversion(node: Node) -> UnaryNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());
        UnaryNode::not(input, output)
    }

    fn nonzero_conversion(node: Node) -> NonZeroNode {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());
        let config = nonzero_config(&node);
        NonZeroNode::new(input, output, config)
    }

    fn and_conversion(node: Node) -> BinaryNode {
        let lhs = Type::from(node.inputs.first().unwrap());
        let rhs = Type::from(node.inputs.get(1).unwrap());
        let output = Type::from(node.outputs.first().unwrap());

        BinaryNode::bool_and(lhs, rhs, output)
    }

    fn or_conversion(node: Node) -> BinaryNode {
        let lhs = Type::from(node.inputs.first().unwrap());
        let rhs = Type::from(node.inputs.get(1).unwrap());
        let output = Type::from(node.outputs.first().unwrap());

        BinaryNode::bool_or(lhs, rhs, output)
    }

    fn xor_conversion(node: Node) -> BinaryNode {
        let lhs = Type::from(node.inputs.first().unwrap());
        let rhs = Type::from(node.inputs.get(1).unwrap());
        let output = Type::from(node.outputs.first().unwrap());

        BinaryNode::bool_xor(lhs, rhs, output)
    }

    fn greater_conversion(node: Node) -> BinaryNode {
        let lhs = Type::from(node.inputs.first().unwrap());
        let rhs = Type::from(node.inputs.get(1).unwrap());
        let output = Type::from(node.outputs.first().unwrap());
        BinaryNode::greater(lhs, rhs, output)
    }

    fn less_conversion(node: Node) -> BinaryNode {
        let lhs = Type::from(node.inputs.first().unwrap());
        let rhs = Type::from(node.inputs.get(1).unwrap());
        let output = Type::from(node.outputs.first().unwrap());
        BinaryNode::lower(lhs, rhs, output)
    }

    fn greater_or_equal_conversion(node: Node) -> BinaryNode {
        let lhs = Type::from(node.inputs.first().unwrap());
        let rhs = Type::from(node.inputs.get(1).unwrap());
        let output = Type::from(node.outputs.first().unwrap());
        BinaryNode::greater_equal(lhs, rhs, output)
    }

    fn less_or_equal_conversion(node: Node) -> BinaryNode {
        let lhs = Type::from(node.inputs.first().unwrap());
        let rhs = Type::from(node.inputs.get(1).unwrap());
        let output = Type::from(node.outputs.first().unwrap());
        BinaryNode::lower_equal(lhs, rhs, output)
    }

    fn pad_conversion(node: Node) -> PadNode {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());
        let config = pad_config(&node);

        PadNode::new(input, output, config)
    }

    fn pow_conversion(node: Node) -> BinaryNode {
        let lhs = Type::from(node.inputs.first().unwrap());
        let rhs = Type::from(node.inputs.get(1).unwrap());
        let output = Type::from(node.outputs.first().unwrap());
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

    fn sign_conversion(node: Node) -> UnaryNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());
        UnaryNode::sign(input, output)
    }

    fn squeeze_conversion(node: Node) -> SqueezeNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());
        let axes = squeeze_config(&node);

        SqueezeNode::new(input, output, axes)
    }

    fn tile_conversion(node: Node) -> TileNode {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());
        let config = tile_config(&node);

        TileNode::new(input, output, config)
    }

    fn top_k_conversion(node: Node) -> TopKNode {
        // Inputs
        let input = TensorType::from(node.inputs.first().unwrap());

        // Outputs
        let outputs = node.outputs.iter().map(TensorType::from).collect();
        let config = top_k_config(&node);

        TopKNode::new(input, outputs, config)
    }

    fn trilu_conversion(node: Node) -> TriluNode {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());
        let config = trilu_config(&node);
        TriluNode::new(input, output, config)
    }

    fn split_conversion(node: Node) -> SplitNode {
        let input = TensorType::from(node.inputs.first().unwrap());
        let outputs = node.outputs.iter().map(TensorType::from).collect();
        let config = split_config(&node);

        SplitNode::new(input, outputs, config)
    }

    fn one_hot_conversion(node: Node) -> OneHotNode {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());
        let values_type = TensorType::from(node.inputs.get(2).unwrap());

        let (num_classes, values, axis) = one_hot_config(&node);
        OneHotNode::new(input, output, num_classes, values, values_type, axis)
    }

    fn floor_conversion(node: Node) -> FloorNode {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());

        FloorNode::new(input, output)
    }

    fn ceil_conversion(node: Node) -> CeilNode {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());

        CeilNode::new(input, output)
    }

    fn round_conversion(node: Node) -> RoundNode {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());

        RoundNode::new(input, output)
    }

    fn gemm_conversion(node: Node) -> GemmNode {
        let a = TensorType::from(node.inputs.first().unwrap());
        let b = TensorType::from(node.inputs.get(1).unwrap());
        let c = node.inputs.get(2).map(Type::from);
        let output = TensorType::from(node.outputs.first().unwrap());
        let (alpha, beta, trans_a, trans_b) = gemm_config(&node);
        GemmNode::new(a, b, c, output, alpha, beta, trans_a, trans_b)
    }

    fn is_inf_conversion(node: Node) -> UnaryNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());
        let config = is_inf_config(&node);
        UnaryNode::is_inf(input, output, config)
    }

    fn is_nan_conversion(node: Node) -> UnaryNode {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());
        UnaryNode::is_nan(input, output)
    }

    fn attention_conversion(node: Node) -> AttentionNode {
        let q = TensorType::from(node.inputs.first().unwrap());
        let k = TensorType::from(node.inputs.get(1).unwrap());
        let v = TensorType::from(node.inputs.get(2).unwrap());
        let attn_mask = node.inputs.get(3).map(TensorType::from);
        let past_key = node.inputs.get(4).map(TensorType::from);
        let past_value = node.inputs.get(5).map(TensorType::from);
        let y = TensorType::from(node.outputs.first().unwrap());
        let present_key = node.outputs.get(1).map(TensorType::from);
        let present_value = node.outputs.get(2).map(TensorType::from);
        let qk_matmul_output = node.outputs.get(3).map(TensorType::from);
        let config = attention_config(&node);

        AttentionNode::new(
            AttentionNodeInputs::new(q, k, v, attn_mask, past_key, past_value),
            AttentionNodeOutputs::new(y, present_key, present_value, qk_matmul_output),
            config,
        )
    }
}

/// Extract data from node states and convert it to `TensorData`.
///
/// # Arguments
///
/// * `input_index` - The index of the input originally from input.
/// * `node` - The node where value are stored.
#[track_caller]
fn extract_data_serialize<E: Element>(input_index: usize, node: &Node) -> Option<TensorData> {
    if node.inputs.is_empty() {
        return None;
    }

    let input = node.inputs.get(input_index);
    input?;
    let input = input.unwrap();
    input.value.as_ref()?;
    let ty = input.ty.clone();

    match ty {
        ArgType::Tensor(_) => {
            let value = input.value.as_ref().expect("Value to be provided.");

            Some(serialize_data::<E>(value.data.clone(), value.shape.clone()))
        }
        _ => panic!("Unsupported serialization type"),
    }
}

/// Convert data to `TensorData`.
fn serialize_data<E: Element>(data: Data, shape: Vec<usize>) -> TensorData {
    match data {
        Data::Float16s(val) => TensorData::new(val, shape).convert::<E>(),
        Data::Float32s(val) => TensorData::new(val, shape).convert::<E>(),
        Data::Float64s(val) => TensorData::new(val, shape).convert::<E>(),
        Data::Int32s(val) => TensorData::new(val, shape).convert::<E>(),
        Data::Int64s(val) => TensorData::new(val, shape).convert::<E>(),
        _ => panic!("Unsupported tensor element type"),
    }
}

/// Convert boolean data to `TensorData`.
fn serialize_bool_data(data: Data, shape: Vec<usize>) -> TensorData {
    match data {
        Data::Bools(val) => TensorData::new(val, shape),
        _ => panic!("Expected boolean data for serialize_bool_data"),
    }
}

impl From<&onnx_ir::ir::Argument> for TensorType {
    fn from(arg: &onnx_ir::ir::Argument) -> Self {
        use onnx_ir::ir::{ArgType, TensorType as OnnxTensorType};

        match &arg.ty {
            ArgType::Tensor(OnnxTensorType {
                elem_type, rank, ..
            }) => tensor_type_from_elem_and_rank(arg.name.clone(), elem_type, *rank),
            ArgType::Scalar(elem_type) => {
                // Represent scalar as rank-0 tensor type of the appropriate kind
                tensor_type_from_elem_and_rank(arg.name.clone(), elem_type, 0)
            }
            ArgType::Shape(_) => panic!("Cannot convert Shape to Burn TensorType"),
        }
    }
}
impl From<&OnnxArgument> for Type {
    fn from(arg: &OnnxArgument) -> Self {
        match &arg.ty {
            ArgType::Tensor(tensor) => {
                // Treat tensor with rank 0 as scalar
                if tensor.rank == 0 {
                    Type::Scalar(ScalarType::new(
                        arg.name.clone(),
                        ScalarKind::from(&tensor.elem_type),
                    ))
                } else {
                    let kind: TensorKind = tensor.elem_type.clone().into();
                    let rank = tensor.rank;
                    let name = arg.name.clone();
                    Type::Tensor(TensorType::new(name, rank, kind))
                }
            }

            ArgType::Scalar(elem_type) => {
                Type::Scalar(ScalarType::new(arg.name.clone(), elem_type.into()))
            }
            ArgType::Shape(rank) => Type::Shape(ShapeType::new(arg.name.clone(), *rank)),
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
            ElementType::Uint16 => ScalarKind::Int32,
            ElementType::Int8 | ElementType::Uint8 => ScalarKind::Int32,
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
            ElementType::Int8 | ElementType::Uint8 => TensorKind::Int,
            ElementType::Bool => TensorKind::Bool,
            _ => panic!("Unsupported tensor type"),
        }
    }
}

fn tensor_type_from_elem_and_rank(name: String, elem: &ElementType, rank: usize) -> TensorType {
    match elem {
        ElementType::Uint8
        | ElementType::Int8
        | ElementType::Uint16
        | ElementType::Int32
        | ElementType::Int64 => TensorType::new(name, rank, TensorKind::Int),

        ElementType::Float16 | ElementType::Float32 | ElementType::Float64 => {
            // If you have TensorType::new_float, use that; otherwise:
            // TensorType::new(name, rank, TensorKind::Float)
            TensorType::new(name, rank, TensorKind::Float)
        }

        ElementType::Bool => TensorType::new(name, rank, TensorKind::Bool),

        ElementType::String => {
            panic!("String element type cannot be converted to Burn TensorType")
        }
    }
}
