use std::{
    env,
    fs::{self, create_dir_all},
    path::{Path, PathBuf},
};

use burn::{
    record::{FullPrecisionSettings, PrecisionSettings},
    tensor::{DataSerialize, Element},
};

use crate::{
    burn::{
        graph::BurnGraph,
        node::{
            batch_norm::BatchNormNode, conv2d::Conv2dNode, flatten::FlattenNode,
            linear::LinearNode, log_softmax::LogSoftmaxNode, matmul::MatmulNode, relu::ReLUNode,
        },
        TensorType,
    },
    format_tokens,
    onnx::{
        ir::{Node, NodeType},
        op_configuration::{
            batch_norm_config, conv2d_config, flatten_config, linear_config, log_softmax_config,
        },
    },
};

use super::{
    from_onnx::parse_onnx,
    ir::{ArgType, Argument, ONNXGraph, Tensor, TensorData},
};

/// Generate code and states from `.onnx` files and save them to the `out_dir`.
#[derive(Debug, Default)]
pub struct ModelGen {
    out_dir: Option<PathBuf>,
    /// List of onnx files to generate source code from.
    inputs: Vec<PathBuf>,
    development: bool,
}

impl ModelGen {
    pub fn new() -> Self {
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

    /// Run code generation.
    fn run(&self, is_build_script: bool) {
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

        create_dir_all(&out_dir).unwrap();

        for input in self.inputs.iter() {
            let file_name = input.file_stem().unwrap();
            let out_file: PathBuf = out_dir.join(file_name);

            Self::generate_model(self.development, input, out_file);
        }
    }

    /// Generate model source code and model state.
    fn generate_model(development: bool, input: &PathBuf, out_file: PathBuf) {
        let graph = parse_onnx(input.as_ref());

        if development {
            // export the graph
            let debug_graph = format!("{:#?}", graph);
            fs::write(out_file.with_extension("graph.txt"), debug_graph).unwrap();
        }

        let graph = graph
            .into_burn::<FullPrecisionSettings>()
            .with_record(
                out_file.clone(),
                development,
                "burn::record::FullPrecisionSettings",
            )
            .with_new_fn(true)
            .with_blank_space(true)
            .with_top_comment(Some(format!(
                "Generated from ONNX {input:?} by burn-import"
            )));

        let code_str = format_tokens(graph.codegen());
        fs::write(out_file.with_extension("rs"), code_str).unwrap();
    }
}

impl ONNXGraph {
    pub fn into_burn<PS: PrecisionSettings + 'static>(self) -> BurnGraph<PS> {
        let mut graph = BurnGraph::<PS>::default();

        for node in self.nodes {
            match node.node_type {
                NodeType::Conv2d => graph.register(Self::conv2d_conversion::<PS>(node)),
                NodeType::MatMul => graph.register(Self::matmul_conversion(node)),
                NodeType::Linear => graph.register(Self::linear_conversion::<PS>(node)),
                NodeType::BatchNormalization => {
                    graph.register(Self::batch_norm_conversion::<PS>(node))
                }
                NodeType::Relu => graph.register(Self::relu_conversion(node)),
                NodeType::Flatten => graph.register(Self::flatten_conversion(node)),
                NodeType::LogSoftmax => graph.register(Self::log_softmax_conversion(node)),
                _ => panic!("Unsupported node conversion {}", node.node_type),
            }
        }

        graph
    }

    fn matmul_conversion(node: Node) -> MatmulNode {
        let lhs = node.inputs.get(0).unwrap().to_tensor_type();
        let rhs = node.inputs.get(1).unwrap().to_tensor_type();
        let output = node.outputs.get(0).unwrap().to_tensor_type();

        MatmulNode::new(lhs, rhs, output)
    }

    fn relu_conversion(node: Node) -> ReLUNode {
        let input = node.inputs.get(0).unwrap().to_tensor_type();
        let output = node.outputs.get(0).unwrap().to_tensor_type();

        ReLUNode::new(input, output)
    }

    fn flatten_conversion(node: Node) -> FlattenNode {
        let input = node.inputs.get(0).unwrap().to_tensor_type();
        let output = node.outputs.get(0).unwrap().to_tensor_type();
        let (start_dim, end_dim) = flatten_config(&node);

        FlattenNode::new(input, output, start_dim, end_dim)
    }

    fn log_softmax_conversion(node: Node) -> LogSoftmaxNode {
        let input = node.inputs.get(0).unwrap().to_tensor_type();
        let output = node.outputs.get(0).unwrap().to_tensor_type();
        let dim = log_softmax_config(&node);

        LogSoftmaxNode::new(input, output, dim)
    }

    fn linear_conversion<PS: PrecisionSettings>(mut node: Node) -> LinearNode<PS> {
        let name = &node.name;
        let input = node.inputs.get(0).unwrap().to_tensor_type();
        let output = node.outputs.get(0).unwrap().to_tensor_type();
        let config = linear_config(&node);

        let bias = node.initializers.len() == 2;
        let weight = node
            .initializers
            .remove(0)
            .arg_type
            .unwrap()
            .into_data_serialize::<PS::FloatElem>();

        let bias = match bias {
            true => Some(
                node.initializers
                    .remove(0)
                    .arg_type
                    .unwrap()
                    .into_data_serialize::<PS::FloatElem>(),
            ),
            false => None,
        };

        LinearNode::new(name, input, output, weight, bias, config)
    }

    fn batch_norm_conversion<PS: PrecisionSettings>(mut node: Node) -> BatchNormNode<PS> {
        let config = batch_norm_config(&node);
        let input = node.inputs.get(0).unwrap().to_tensor_type();
        let output = node.outputs.get(0).unwrap().to_tensor_type();
        let dim = input.dim - 2;

        let gamma =
            extract_next_data_serialize::<PS::FloatElem>(&mut node).expect("Gamma is required");
        let beta =
            extract_next_data_serialize::<PS::FloatElem>(&mut node).expect("Gamma is required");
        let running_mean = extract_next_data_serialize::<PS::FloatElem>(&mut node)
            .expect("Running mean is required");
        let running_var = extract_next_data_serialize::<PS::FloatElem>(&mut node)
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

    fn conv2d_conversion<PS: PrecisionSettings>(mut node: Node) -> Conv2dNode<PS> {
        let input = node.inputs.get(0).unwrap().to_tensor_type();
        let output = node.outputs.get(0).unwrap().to_tensor_type();
        let config = conv2d_config(&node);

        let bias = node.initializers.len() == 2;
        let weight = extract_next_data_serialize::<PS::FloatElem>(&mut node).unwrap();
        let bias = match bias {
            true => Some(extract_next_data_serialize::<PS::FloatElem>(&mut node)).unwrap(),
            false => None,
        };

        let name = &node.name;
        Conv2dNode::<PS>::new(name, input, output, weight, bias, config)
    }
}

fn extract_next_data_serialize<E: Element>(node: &mut Node) -> Option<DataSerialize<E>> {
    if node.initializers.is_empty() {
        return None;
    }

    node.initializers
        .remove(0)
        .arg_type
        .map(|arg| arg.into_data_serialize::<E>())
}

impl ArgType {
    pub fn into_data_serialize<E: Element>(self) -> DataSerialize<E> {
        match self {
            ArgType::Tensor(tensor) => tensor.into_data_serialize(),
        }
    }
}

impl Argument {
    pub fn to_tensor_type(&self) -> TensorType {
        match self.arg_type.as_ref().expect("Tensor arg type") {
            ArgType::Tensor(tensor) => TensorType::new(self.name.clone(), tensor.shape.len()),
        }
    }
}

impl Tensor {
    pub fn into_data_serialize<E: Element>(self) -> DataSerialize<E> {
        let data = self.data.expect("Data to be provided.");

        match data {
            TensorData::Float16(val) => DataSerialize::new(val, self.shape).convert(),
            TensorData::Float32(val) => DataSerialize::new(val, self.shape).convert(),
            TensorData::Float64(val) => DataSerialize::new(val, self.shape).convert(),
            TensorData::Int32(val) => DataSerialize::new(val, self.shape).convert(),
            TensorData::Int64(val) => DataSerialize::new(val, self.shape).convert(),
            TensorData::String(_) => panic!("String tensor unsuported"),
        }
    }
}
