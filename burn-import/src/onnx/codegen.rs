use std::{
    collections::HashSet,
    env,
    fs::{self, create_dir_all},
    path::{Path, PathBuf},
};

use burn::nn::conv::Conv2dPaddingConfig;
use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::{Ident, Type};

use crate::onnx::{
    ir::{ArgType, Node, NodeType},
    op_configuration::{conv2d_config, flatten_config, linear_config, log_softmax_config},
};

use super::{convert::parse_onnx, ir::Graph};

use rust_format::{Config, Edition, Formatter, PostProcess, RustFmt};

/// Code generation for onnx files.
#[derive(Debug, Default)]
pub struct ModelCodeGen {
    out_dir: Option<PathBuf>,

    /// List of onnx files to generate source code from.
    inputs: Vec<PathBuf>,
}

/// Generate code from `.onnx` files and save it to the `out_dir`.
impl ModelCodeGen {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set output directory.
    pub fn out_dir(&mut self, out_dir: &str) -> &mut Self {
        let cargo_out_dir = env::var("OUT_DIR").expect("OUT_DIR env is not set");
        let mut path = PathBuf::from(cargo_out_dir);

        // Append the out_dir to the cargo_out_dir
        path.push(Path::new(out_dir));

        self.out_dir = Some(path);
        self
    }

    /// Add input file.
    pub fn input(&mut self, input: &str) -> &mut Self {
        self.inputs.push(input.into());
        self
    }

    /// Run code generation.
    ///
    /// This function is intended to be called from `build.rs` script.
    pub fn run_from_script(&self) {
        self.run();
    }

    /// Run code generation.
    pub fn run(&self) {
        let config = Config::new_str()
            .post_proc(PostProcess::ReplaceMarkersAndDocBlocks)
            .edition(Edition::Rust2021);

        let rust_formatter = RustFmt::from_config(config);

        let out_dir = self.out_dir.as_ref().expect("out_dir is not set");
        create_dir_all(out_dir).unwrap();

        for input in self.inputs.iter() {
            let file_name = input.file_stem().unwrap();
            let out_file = out_dir.join(file_name);
            let out_file = out_file.with_extension("rs");

            let model = ModelSourceCode::new(input);
            let code_str = rust_formatter.format_tokens(model.body()).unwrap();

            fs::write(out_file, code_str).unwrap();
        }
    }
}

/// A model that can be used to generate code
#[derive(Debug, Clone)]
pub struct ModelSourceCode {
    onnx_path: PathBuf,
    pub graph: Graph,
}

impl ModelSourceCode {
    /// Create a new model from the onnx file
    pub fn new<P: AsRef<Path>>(onnx_path: P) -> Self {
        let graph = parse_onnx(onnx_path.as_ref());
        Self {
            onnx_path: onnx_path.as_ref().to_path_buf(),
            graph,
        }
    }

    /// Generates source code for the model
    pub fn body(&self) -> TokenStream {
        let input = "Model"; // TODO make this a parameter
        let input = Ident::new(input, Span::call_site());

        let declaration = self.declaration(&input);

        let file_path = self.onnx_path.to_str().unwrap();

        let top_file_comment = format!("Generated from {file_path} by burn-import");

        let mut imports: HashSet<String> = HashSet::new();

        let implementation = self.implementation(&mut imports);

        let import_statements = self.import_statements(&imports);

        let shape_constants = self.shape_constants();

        //TODO print out the old -> new name mapping
        quote! {
            _comment_!(#top_file_comment);
            _blank_!();
            _blank_!();
            #import_statements
            _blank_!();
            #shape_constants
            _blank_!();
            #declaration
            _blank_!();
            #[allow(dead_code)]
            #[allow(clippy::new_without_default)]
            #[allow(clippy::let_and_return)]
            #implementation

        }
    }

    fn shape_constants(&self) -> TokenStream {
        let input_constants = self.graph.inputs.iter().enumerate().map(|(i, input)| {
            let name = format!("INPUT{}_SHAPE", i + 1);
            let name = Ident::new(&name, Span::call_site());
            let ArgType::Tensor(tensor) = input.clone().arg_type.unwrap();
            let dims = tensor.shape;
            let dims_count = dims.len();
            quote! {
                pub const #name: [usize; #dims_count] = [#(#dims),*];
            }
        });

        let output_constants = self.graph.outputs.iter().enumerate().map(|(i, input)| {
            let name = format!("OUTPUT{}_SHAPE", i + 1);
            let name = Ident::new(&name, Span::call_site());
            let ArgType::Tensor(tensor) = input.clone().arg_type.unwrap();
            let dims = tensor.shape;
            let dims_count = dims.len();
            quote! {
                pub const #name: [usize; #dims_count] = [#(#dims),*];
            }
        });

        quote! {
            #(#input_constants)*
            #(#output_constants)*
        }
    }

    /// Generates import statements for the model
    fn import_statements(&self, imports: &HashSet<String>) -> TokenStream {
        let mut import_tokens = vec![];

        for import in imports.iter() {
            let path: syn::Path =
                syn::parse_str(import).expect("Unable to parse input string as a path");

            import_tokens.push(quote! { #path });
        }

        quote! {
            use burn::{
                module::Module,
                nn,
                tensor::{backend::Backend, Tensor},
            };

            #(use #import_tokens;)*
        }
    }

    /// Generates the declaration portion of the source code for the model
    fn declaration(&self, name: &Ident) -> TokenStream {
        let fields = self.declaration_fields();

        let mut field_names = vec![];
        let mut field_types = vec![];

        for (field_name, field_type) in fields.iter() {
            field_names.push(field_name);
            field_types.push(field_type);
        }

        quote! {
            // TODO add documentation
            #[doc = "This is a generated model from an ONNX file"]
            #[derive(Module, Debug)]
            pub struct #name<B: Backend> {
                #(
                   #field_names: #field_types,
                )*
            }

        }
    }

    /// Model implementation code
    fn implementation(&self, imports: &mut HashSet<String>) -> TokenStream {
        let forward_method = self.forward_method(imports);

        let new_method = self.new_method();

        quote! {
            impl<B: Backend> Model<B> {
                #new_method
                #forward_method
            }
        }
    }

    /// Generates the new method for the model
    fn forward_method(&self, imports: &mut HashSet<String>) -> TokenStream {
        let inputs = self.forward_signature_input();
        let return_type = self.forward_signature_return();
        let results = self.forward_method_results();

        let mut call_nodes: Vec<TokenStream> = vec![];

        for node in self.graph.nodes.iter() {
            if node.is_stateful {
                call_nodes.push(Self::node_call_stateful(node));
            } else {
                call_nodes.push(Self::node_call_stateless(node, imports));
            }
        }

        quote! {
            pub fn forward(&self, #(#inputs,)*) -> #return_type {
                #(#call_nodes)*
                #results
            }
        }
    }

    /// Generates source code for the stateful node calls, i.e. conv, dropout, etc.
    fn node_call_stateful(node: &Node) -> TokenStream {
        if !node.is_stateful {
            panic!("Node must be stateful");
        }

        let name = Ident::new(&node.name, Span::call_site());

        let mut inputs = vec![];

        for input in node.inputs.iter() {
            let name = Ident::new(&input.name, Span::call_site());
            inputs.push(quote! {
                #name
            });
        }

        let mut outputs = vec![];

        for output in node.outputs.iter() {
            let name = Ident::new(&output.name, Span::call_site());
            outputs.push(quote! {
                #name
            });
        }

        if outputs.len() == 1 {
            let output = outputs.pop().unwrap();
            quote! {
                let #output = self.#name.forward(#(#inputs,)*);
            }
        } else {
            quote! {
                let (#(#outputs,)*) = self.#name.forward(#(#inputs,)*);
            }
        }
    }

    /// Generates source code for the forward method results
    fn forward_method_results(&self) -> TokenStream {
        let mut outputs = vec![];
        for output in self.graph.outputs.iter() {
            let name = Ident::new(&output.name, Span::call_site());
            outputs.push(quote! {
                #name
            });
        }
        if outputs.len() == 1 {
            let output = outputs.pop().unwrap();
            quote! {
                #output
            }
        } else {
            quote! {
                (#(#outputs,)*)
            }
        }
    }

    /// Generates source code for the stateless node calls, i.e. add, mul, etc.
    fn node_call_stateless(node: &Node, imports: &mut HashSet<String>) -> TokenStream {
        if node.is_stateful {
            panic!("Node must be stateless");
        }

        let mut inputs = vec![];

        for input in node.inputs.iter() {
            let name = Ident::new(&input.name, Span::call_site());
            inputs.push(quote! {
                #name
            });
        }

        let mut outputs = vec![];

        for output in node.outputs.iter() {
            let name = Ident::new(&output.name, Span::call_site());
            outputs.push(quote! {
                #name
            });
        }

        let rhs = Self::node_call_stateless_rhs(node, imports);

        if outputs.len() == 1 {
            let output = outputs.pop().unwrap();
            quote! {
                let #output = #rhs;
            }
        } else {
            quote! {
                let (#(#outputs,)*) = #rhs;
            }
        }
    }

    /// Generates source code for the right hand side stateless node calls, i.e. add, relu, etc.
    fn node_call_stateless_rhs(node: &Node, imports: &mut HashSet<String>) -> TokenStream {
        let mut inputs = vec![];

        for input in node.inputs.iter() {
            let name = Ident::new(&input.name, Span::call_site());
            inputs.push(quote! {
                #name
            });
        }

        let input1 = inputs.pop().unwrap();

        match node.node_type {
            NodeType::Relu => {
                imports.insert("burn::tensor::activation::relu".to_string());

                quote! { relu(#input1) }
            }
            NodeType::LogSoftmax => {
                imports.insert("burn::tensor::activation::log_softmax".to_string());
                let dim = log_softmax_config(node);

                quote! { log_softmax(#input1, #dim) }
            }
            NodeType::Flatten => {
                let (start_dim, end_dim) = flatten_config(node);

                quote! { #input1.flatten(#start_dim, #end_dim) }
            }
            _ => quote! {},
        }
    }

    /// Generates the forward method signature
    fn forward_signature_input(&self) -> Vec<TokenStream> {
        let mut fields = vec![];

        for input in self.graph.inputs.iter() {
            let name = Ident::new(&input.name, Span::call_site());

            let ty = match input.arg_type.as_ref().unwrap() {
                ArgType::Tensor(tensor) => {
                    let d = &tensor.shape.len();
                    syn::parse_str::<Type>(format!("Tensor<B, {d}>").as_str()).unwrap()
                }
            };

            fields.push(quote! {
                #name: #ty
            });
        }
        fields
    }

    /// Generates the forward method return signature
    fn forward_signature_return(&self) -> TokenStream {
        let mut field_types = vec![];

        for output in self.graph.outputs.iter() {
            let ty = match output.arg_type.as_ref().unwrap() {
                ArgType::Tensor(tensor) => {
                    let d = &tensor.shape.len();
                    syn::parse_str::<Type>(format!("Tensor<B, {d}>").as_str()).unwrap()
                }
            };

            field_types.push(ty);
        }

        if field_types.len() == 1 {
            // Return one output
            quote! {
                #(
                    #field_types
                 )*
            }
        } else {
            // Return a tuple of the outputs
            quote! {
                (#(
                    #field_types,
                 )*)
            }
        }
    }

    /// Generates source code for the initialization method
    fn new_method(&self) -> TokenStream {
        let initialization_fields = self.initialization_fields();

        let field_names = self.graph.nodes.iter().filter(|x| x.is_stateful).map(|x| {
            let name = Ident::new(&x.name, Span::call_site());
            quote! {
                #name
            }
        });

        quote! {
            pub fn new() -> Self {
                #(
                    #initialization_fields
                )*

                Self {
                    #(
                        #field_names
                    ),*
                }
            }
        }
    }

    /// Get the fields for the declaration of the model
    fn declaration_fields(&self) -> Vec<(Ident, Type)> {
        let mut fields = vec![];

        for node in self.graph.nodes.iter().filter(|x| x.is_stateful) {
            let node_type = match node.node_type {
                NodeType::Conv1d => syn::parse_str::<Type>("nn::conv::Conv1d<B>").unwrap(),
                NodeType::Conv2d => syn::parse_str::<Type>("nn::conv::Conv2d<B>").unwrap(),
                NodeType::Linear => syn::parse_str::<Type>("nn::Linear<B>").unwrap(),
                _ => {
                    todo!("Node type not implemented: {:?}", node.node_type)
                }
            };

            let node_name = Ident::new(&node.name, Span::call_site());

            fields.push((node_name, node_type));
        }

        fields
    }

    /// Generates source code for the initialization method
    fn initialization_fields(&self) -> Vec<TokenStream> {
        let mut fields = vec![];

        for node in self.graph.nodes.iter().filter(|x| x.is_stateful) {
            let init_code = match node.node_type {
                NodeType::Conv2d => conv2d_init(node),
                NodeType::Linear => linear_init(node),
                _ => {
                    todo!("Node type not implemented: {:?}", node.node_type)
                }
            };

            fields.push(init_code);
        }

        fields
    }
}

/// Generates source code for the initialization of a Conv2d node
fn conv2d_init(node: &Node) -> TokenStream {
    let node_name = Ident::new(&node.name, Span::call_site());

    let config = conv2d_config(node);

    let channel_in = config.channels[0];
    let channel_out = config.channels[1];
    let kernel_size_0 = config.kernel_size[0];
    let kernel_size_1 = config.kernel_size[1];
    let bias = config.bias;

    let padding = match config.padding {
        Conv2dPaddingConfig::Valid => quote! { nn::conv::Conv2dPaddingConfig::Valid },
        Conv2dPaddingConfig::Same => quote! { nn::conv::Conv2dPaddingConfig::Same },
        _ => todo!("Padding ({:?}) not implemented", config.padding),
    };

    quote! {
        let #node_name = nn::conv::Conv2dConfig::new([#channel_in, #channel_out], [#kernel_size_0, #kernel_size_1])
            .with_padding(#padding)
            .with_bias(#bias)
            .init();

    }
}

/// Generates source code for the initialization of a Linear node
fn linear_init(node: &Node) -> TokenStream {
    let node_name = Ident::new(&node.name, Span::call_site());
    let config = linear_config(node);

    let bias = config.bias;
    let input_size = config.d_input;
    let output_size = config.d_output;

    quote! {
        let #node_name = nn::LinearConfig::new(#input_size, #output_size)
            .with_bias(#bias)
            .init();

    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;
    use rust_format::{Config, Edition, Formatter, PostProcess, RustFmt};

    #[fixture]
    pub fn model() -> ModelSourceCode {
        ModelSourceCode::new("tests/onnx/mnist.onnx")
    }

    #[rstest]
    fn print(model: ModelSourceCode) {
        let config = Config::new_str()
            .post_proc(PostProcess::ReplaceMarkersAndDocBlocks)
            .edition(Edition::Rust2021);

        let rustfmt = RustFmt::from_config(config);

        let _gen_str = rustfmt.format_tokens(model.body()).unwrap();

        // TODO compare the result with the expected output
    }
}
