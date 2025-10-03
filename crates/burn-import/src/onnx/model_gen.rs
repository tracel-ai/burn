use std::{
    env,
    fs::{self, create_dir_all},
    path::{Path, PathBuf},
};

use burn::record::{
    DoublePrecisionSettings, FullPrecisionSettings, HalfPrecisionSettings, PrecisionSettings,
};

use crate::{
    burn::{graph::BurnGraph, node::try_convert_onnx_node},
    format_tokens,
    logger::init_log,
};

use onnx_ir::{ir::OnnxGraph, parse_onnx};

pub use crate::burn::graph::RecordType;

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
            // Try registry-based conversion
            if let Some(burn_node) = try_convert_onnx_node::<PS>(node.clone()) {
                graph.register(burn_node);
            } else {
                // Unsupported node type
                unsupported_ops.push(node.node_type);
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
}
