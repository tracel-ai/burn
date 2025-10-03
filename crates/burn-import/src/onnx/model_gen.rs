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

        let out_dir = self.get_output_directory(is_build_script);
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

    /// Get the output directory path based on whether this is a build script or CLI invocation.
    fn get_output_directory(&self, is_build_script: bool) -> PathBuf {
        if is_build_script {
            let cargo_out_dir = env::var("OUT_DIR").expect("OUT_DIR env is not set");
            let mut path = PathBuf::from(cargo_out_dir);
            // Append the out_dir to the cargo_out_dir
            path.push(self.out_dir.as_ref().unwrap());
            path
        } else {
            self.out_dir.as_ref().expect("out_dir is not set").clone()
        }
    }

    /// Generate model source code and model state.
    fn generate_model(&self, input: &PathBuf, out_file: PathBuf) {
        log::info!("Generating model from {input:?}");
        log::debug!("Development mode: {:?}", self.development);
        log::debug!("Output file: {out_file:?}");

        let graph = parse_onnx(input.as_ref());

        if self.development {
            self.write_debug_file(&out_file, "onnx.txt", &graph);
        }

        let graph = ParsedOnnxGraph(graph);

        if self.development {
            self.write_debug_file(&out_file, "graph.txt", &graph);
        }

        let top_comment = Some(format!("Generated from ONNX {input:?} by burn-import"));

        let code = self.generate_code_with_precision(graph, &out_file, top_comment);

        let code_str = format_tokens(code);
        let source_code_file = out_file.with_extension("rs");
        log::info!("Writing source code to {}", source_code_file.display());
        fs::write(source_code_file, code_str).unwrap();

        log::info!("Model generated");
    }

    /// Write debug file in development mode.
    fn write_debug_file<T: std::fmt::Debug>(
        &self,
        out_file: &PathBuf,
        extension: &str,
        content: &T,
    ) {
        let debug_content = format!("{content:#?}");
        let debug_file = out_file.with_extension(extension);
        log::debug!("Writing debug file: {debug_file:?}");
        fs::write(debug_file, debug_content).unwrap();
    }

    /// Generate code with appropriate precision settings.
    fn generate_code_with_precision(
        &self,
        graph: ParsedOnnxGraph,
        out_file: &PathBuf,
        top_comment: Option<String>,
    ) -> proc_macro2::TokenStream {
        let blank_space = true;

        if self.double_precision {
            self.generate_burn_graph::<DoublePrecisionSettings>(
                graph,
                out_file,
                blank_space,
                top_comment,
            )
        } else if self.half_precision {
            self.generate_burn_graph::<HalfPrecisionSettings>(
                graph,
                out_file,
                blank_space,
                top_comment,
            )
        } else {
            self.generate_burn_graph::<FullPrecisionSettings>(
                graph,
                out_file,
                blank_space,
                top_comment,
            )
        }
    }

    /// Generate BurnGraph with specified precision settings and codegen.
    fn generate_burn_graph<PS: PrecisionSettings + 'static>(
        &self,
        graph: ParsedOnnxGraph,
        out_file: &PathBuf,
        blank_space: bool,
        top_comment: Option<String>,
    ) -> proc_macro2::TokenStream {
        graph
            .into_burn::<PS>()
            .with_record(out_file.clone(), self.record_type, self.embed_states)
            .with_blank_space(blank_space)
            .with_top_comment(top_comment)
            .codegen()
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

        // Extract input and output names
        let input_names: Vec<_> = self
            .0
            .inputs
            .iter()
            .map(|input| input.name.clone())
            .collect();
        let output_names: Vec<_> = self
            .0
            .outputs
            .iter()
            .map(|output| output.name.clone())
            .collect();

        // Register inputs and outputs with the graph
        graph.register_input_output(input_names, output_names);

        graph
    }
}
