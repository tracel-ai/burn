use super::{BurnImports, Scope};
use crate::burn::node::NodeCodegen;
use burn_store::{BurnpackWriter, TensorSnapshot};
use onnx_ir::{Node, ir::ArgType};
use proc_macro2::TokenStream;
use quote::quote;
use std::{collections::HashMap, path::PathBuf};

/// Burn graph intermediate representation of modules and tensor operations.
#[derive(Default, Debug)]
pub struct BurnGraph {
    nodes: Vec<Node>,
    scope: Scope,
    imports: BurnImports,
    top_comment: Option<String>,
    default: Option<TokenStream>,
    blank_spaces: bool,
    graph_input_args: Vec<onnx_ir::Argument>,
    graph_output_args: Vec<onnx_ir::Argument>,
}

impl BurnGraph {
    /// Register a new operation node into the graph.
    ///
    /// # Notes
    ///
    /// The node must be registered in the same order they will be executed in the forward pass.
    pub fn register(&mut self, node: Node) {
        log::debug!("Registering node => '{}'", node.name());
        self.nodes.push(node);
    }

    /// Save the state of each node in a burnpack file.
    ///
    /// The `Default` trait will be implemented for the generated model, which will load the
    /// burnpack file saved at the provided path.
    ///
    /// # Arguments
    ///
    /// * `out_file` - The path to the burnpack file (without extension).
    /// * `embed_states` - If true, embed the burnpack file in the binary using `include_bytes!`.
    pub fn with_burnpack(mut self, out_file: PathBuf, embed_states: bool) -> Self {
        // Collect all tensor snapshots from nodes
        let snapshots = self.collect_all_snapshots();

        // Write burnpack file
        let burnpack_file = out_file.with_extension("bpk");
        BurnpackWriter::new(snapshots)
            .with_metadata("producer", "burn-import")
            .write_to_file(&burnpack_file)
            .expect("Failed to write burnpack file");

        // Register the loading code
        if embed_states {
            self.register_burnpack_embed(burnpack_file);
        } else {
            self.register_burnpack_file(burnpack_file);
        }

        self
    }

    /// Collect all tensor snapshots from nodes recursively.
    fn collect_all_snapshots(&self) -> Vec<TensorSnapshot> {
        let mut snapshots = Vec::new();
        let mut field_name_counts: HashMap<String, usize> = HashMap::new();

        // Helper to recursively collect snapshots from subgraphs
        fn collect_subgraph_snapshots_recursive(
            subgraph: &onnx_ir::OnnxGraph,
            field_name_counts: &mut HashMap<String, usize>,
            snapshots: &mut Vec<TensorSnapshot>,
        ) {
            for node in &subgraph.nodes {
                if let Some(field) = NodeCodegen::field(node) {
                    let base_name = field.name.to_string();
                    let count = field_name_counts.entry(base_name.clone()).or_insert(0);
                    *count += 1;

                    // Create unique name if needed
                    let unique_name = if *count > 1 {
                        format!("{}_{}", base_name, count)
                    } else {
                        base_name
                    };

                    // Collect snapshots for this node
                    let node_snapshots = NodeCodegen::collect_snapshots(node, &unique_name);
                    snapshots.extend(node_snapshots);
                }

                // Recursively collect from nested If/Loop nodes
                if let Node::If(nested_if_node) = node {
                    collect_subgraph_snapshots_recursive(
                        &nested_if_node.config.then_branch,
                        field_name_counts,
                        snapshots,
                    );
                    collect_subgraph_snapshots_recursive(
                        &nested_if_node.config.else_branch,
                        field_name_counts,
                        snapshots,
                    );
                } else if let Node::Loop(nested_loop_node) = node {
                    collect_subgraph_snapshots_recursive(
                        &nested_loop_node.config.body,
                        field_name_counts,
                        snapshots,
                    );
                }
            }
        }

        // Collect from main graph nodes
        for node in &self.nodes {
            if let Some(field) = NodeCodegen::field(node) {
                let field_name = field.name.to_string();
                let node_snapshots = NodeCodegen::collect_snapshots(node, &field_name);
                snapshots.extend(node_snapshots);
            }

            // Collect from subgraphs in If/Loop nodes
            if let Node::If(if_node) = node {
                collect_subgraph_snapshots_recursive(
                    &if_node.config.then_branch,
                    &mut field_name_counts,
                    &mut snapshots,
                );
                collect_subgraph_snapshots_recursive(
                    &if_node.config.else_branch,
                    &mut field_name_counts,
                    &mut snapshots,
                );
            } else if let Node::Loop(loop_node) = node {
                collect_subgraph_snapshots_recursive(
                    &loop_node.config.body,
                    &mut field_name_counts,
                    &mut snapshots,
                );
            }
        }

        snapshots
    }

    /// Add blank spaces in some places
    ///
    /// # Notes
    ///
    /// It can be problematic when testing.
    pub fn with_blank_space(mut self, blank_spaces: bool) -> Self {
        self.blank_spaces = blank_spaces;
        self
    }

    /// Add a comment at the top of the generated file.
    pub fn with_top_comment(mut self, top_comment: Option<String>) -> Self {
        self.top_comment = top_comment;
        self
    }

    /// Generate tokens representing the graph with Burn modules and tensor operations.
    pub fn codegen(mut self) -> TokenStream {
        self.build_scope();

        self.register_imports();

        let codegen_imports = self.imports.codegen();
        let codegen_struct = self.codegen_struct();
        let codegen_new = self.codegen_new();
        let codegen_forward = self.codegen_forward();

        let maybe_blank = match self.blank_spaces {
            true => quote! {
                _blank_!();
            },
            false => quote! {},
        };
        let codegen_default = match self.default {
            Some(default) => quote! {
                #default
                #maybe_blank
            },
            None => quote! {},
        };

        let maybe_top_file_comment = match self.top_comment {
            Some(comment) => quote! {
                _comment_!(#comment);
            },
            None => quote! {},
        };

        quote! {
            // @generated
            // This file is automatically generated by burn-import

            #maybe_top_file_comment

            #[allow(unused)]
            mod generated {

                #codegen_imports
                #maybe_blank
                #maybe_blank

                #codegen_struct
                #maybe_blank

                #codegen_default

                impl<B: Backend> Model<B> {
                    #codegen_new

                    #maybe_blank

                    #codegen_forward
                }
            }
            pub use generated::*;
        }
    }

    fn register_imports(&mut self) {
        // Register imports from nodes
        self.nodes
            .iter()
            .for_each(|node| NodeCodegen::register_imports(node, &mut self.imports));
    }

    /// Build the scope state to make sure tensor clones are added where needed.
    fn build_scope(&mut self) {
        log::debug!("Building the scope nodes len => '{}'", self.nodes.len());

        // Register graph tensor inputs with 0 as node position
        self.graph_input_args
            .iter()
            .filter(|arg| matches!(arg.ty, ArgType::Tensor(_)))
            .for_each(|arg| {
                self.scope.tensor_register_variable(arg, 0);
            });

        self.nodes
            .iter()
            .enumerate()
            .for_each(|(node_position, node)| {
                // Register tensor outputs as variables
                node.outputs()
                    .iter()
                    .filter(|arg| matches!(arg.ty, ArgType::Tensor(_)))
                    .for_each(|arg| {
                        self.scope.tensor_register_variable(arg, node_position + 1);
                    });
                // Since the graph is guaranteed to be a DAG, we can safely register future uses
                // of the inputs (which are the previous nodes' outputs)
                // Filter to only dynamic/constant inputs (exclude static-only initializers)
                node.inputs()
                    .iter()
                    .filter(|arg| arg.is_dynamic() || arg.is_constant())
                    .filter(|arg| matches!(arg.ty, ArgType::Tensor(_)))
                    .for_each(|arg| self.scope.tensor_register_future_use(arg, node_position));
            });

        // Register graph tensor output with the last node position
        self.graph_output_args
            .iter()
            .filter(|arg| matches!(arg.ty, ArgType::Tensor(_)))
            .for_each(|arg| {
                self.scope.tensor_register_future_use(arg, self.nodes.len());
            });
    }

    fn register_burnpack_file(&mut self, file: PathBuf) {
        self.imports.register("burn_store::BurnpackStore");
        self.imports.register("burn_store::ModuleSnapshot");

        let file = file.to_str().unwrap();
        self.default = Some(quote! {
            _blank_!();
            impl<B: Backend> Default for Model<B> {
                fn default() -> Self {
                    Self::from_file(#file, &Default::default())
                }
            }
            _blank_!();
            impl<B: Backend> Model<B> {
                /// Load model weights from a burnpack file.
                pub fn from_file(file: &str, device: &B::Device) -> Self {
                    let mut model = Self::new(device);
                    let mut store = BurnpackStore::from_file(file);
                    model.load_from(&mut store).expect("Failed to load burnpack file");
                    model
                }
            }
        });
    }

    fn register_burnpack_embed(&mut self, file: PathBuf) {
        self.imports.register("burn_store::BurnpackStore");
        self.imports.register("burn_store::ModuleSnapshot");

        let file = file.to_str().unwrap();
        self.default = Some(quote! {
            _blank_!();
            static EMBEDDED_STATES: &[u8] = include_bytes!(#file);
            _blank_!();
            impl<B: Backend> Default for Model<B> {
                fn default() -> Self {
                    Self::from_embedded(&Default::default())
                }
            }
            _blank_!();
            impl<B: Backend> Model<B> {
                /// Load model weights from embedded burnpack data (zero-copy at store level).
                ///
                /// The embedded data stays in the binary's .rodata section without heap allocation.
                /// Tensor data is sliced directly from the static bytes.
                ///
                /// Note: Some backends (e.g., NdArray) may still copy data internally.
                /// See <https://github.com/tracel-ai/burn/issues/4153> for true backend zero-copy.
                ///
                /// See <https://github.com/tracel-ai/burn/issues/4123>
                pub fn from_embedded(device: &B::Device) -> Self {
                    let mut model = Self::new(device);
                    let mut store = BurnpackStore::from_static(EMBEDDED_STATES);
                    model.load_from(&mut store).expect("Failed to load embedded burnpack");
                    model
                }
            }
        });
    }

    /// Recursively collect all fields from nodes, including subgraph nodes in If/Loop/Scan
    fn collect_all_fields(&self) -> Vec<(proc_macro2::Ident, TokenStream, Option<TokenStream>)> {
        // Track field name usage to make them unique
        let mut field_name_counts: HashMap<String, usize> = HashMap::new();
        let mut all_fields: Vec<(proc_macro2::Ident, TokenStream, Option<TokenStream>)> =
            Vec::new();

        // Helper to recursively collect fields from a subgraph and its nested subgraphs
        fn collect_subgraph_fields_recursive(
            subgraph: &onnx_ir::OnnxGraph,
            field_name_counts: &mut HashMap<String, usize>,
            all_fields: &mut Vec<(proc_macro2::Ident, TokenStream, Option<TokenStream>)>,
        ) {
            for onnx_node in &subgraph.nodes {
                let burn_node = onnx_node;
                // Collect this node's field if it has one
                if let Some(mut field) = NodeCodegen::field(burn_node) {
                    // Make field name unique by appending a counter if needed
                    let base_name = field.name.to_string();
                    let count = field_name_counts.entry(base_name.clone()).or_insert(0);
                    *count += 1;

                    // Only append counter if this name has been seen before
                    if *count > 1 {
                        // Need to create a new renamed field
                        let new_name_str = format!("{}_{}", base_name, count);
                        let new_name =
                            syn::Ident::new(&new_name_str, proc_macro2::Span::call_site());

                        // Update the field name
                        field.name = new_name.clone();

                        // Also need to update field.init to use the renamed variable
                        let init_str = field.init.to_string();
                        let old_let = format!("let {} :", base_name);
                        let new_let = format!("let {} :", new_name_str);
                        let updated_init_str = init_str.replace(&old_let, &new_let);

                        // Also handle "let base_name ="
                        let old_let2 = format!("let {} =", base_name);
                        let new_let2 = format!("let {} =", new_name_str);
                        let updated_init_str = updated_init_str.replace(&old_let2, &new_let2);

                        // Parse back to TokenStream
                        let updated_init: TokenStream = updated_init_str
                            .parse()
                            .unwrap_or_else(|_| field.init.clone());
                        field.init = updated_init;
                    }
                    all_fields.push((field.name.clone(), field.ty.clone(), Some(field.init)));
                }

                // Recursively collect from nested If/Loop nodes
                if let Node::If(nested_if_node) = burn_node {
                    collect_subgraph_fields_recursive(
                        &nested_if_node.config.then_branch,
                        field_name_counts,
                        all_fields,
                    );
                    collect_subgraph_fields_recursive(
                        &nested_if_node.config.else_branch,
                        field_name_counts,
                        all_fields,
                    );
                } else if let Node::Loop(nested_loop_node) = burn_node {
                    collect_subgraph_fields_recursive(
                        &nested_loop_node.config.body,
                        field_name_counts,
                        all_fields,
                    );
                }
            }
        }

        for node in &self.nodes {
            // Collect this node's field if it has one
            if let Some(field) = NodeCodegen::field(node) {
                all_fields.push((field.name, field.ty, Some(field.init)));
            }

            // Recursively collect fields from If/Loop node subgraphs
            // Note: Subgraph fields are NOT deduplicated - each branch has unique fields
            if let Node::If(if_node) = node {
                collect_subgraph_fields_recursive(
                    &if_node.config.then_branch,
                    &mut field_name_counts,
                    &mut all_fields,
                );
                collect_subgraph_fields_recursive(
                    &if_node.config.else_branch,
                    &mut field_name_counts,
                    &mut all_fields,
                );
            } else if let Node::Loop(loop_node) = node {
                collect_subgraph_fields_recursive(
                    &loop_node.config.body,
                    &mut field_name_counts,
                    &mut all_fields,
                );
            }
        }

        all_fields
    }

    fn codegen_struct(&self) -> TokenStream {
        let mut body = quote! {};
        self.collect_all_fields()
            .iter()
            .map(|(name, ty, _)| {
                quote! {
                    #name: #ty,
                }
            })
            .for_each(|code| body.extend(code));

        // Extend with phantom data to avoid unused generic type.
        body.extend(quote! {
            phantom: core::marker::PhantomData<B>,
            device: burn::module::Ignored<B::Device>,
        });

        quote! {
            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                #body
            }
        }
    }

    fn codegen_new(&self) -> TokenStream {
        let mut body = quote! {};
        let all_fields = self.collect_all_fields();

        // Generate field initialization code
        for (_, _, field_init) in &all_fields {
            body.extend(field_init.clone());
        }

        // Collect field names for struct initialization
        let field_names: Vec<_> = all_fields.iter().map(|(name, _, _)| name.clone()).collect();

        quote! {
            #[allow(unused_variables)]
            pub fn new(device: &B::Device) -> Self {
                #body

                Self {
                    #(#field_names,)*
                    phantom: core::marker::PhantomData,
                    device: burn::module::Ignored(device.clone()),
                }
            }
        }
    }

    fn codegen_forward(&mut self) -> TokenStream {
        let input_def = crate::burn::codegen_fn_params(&self.graph_input_args);
        let output_type_def = crate::burn::codegen_return_type(&self.graph_output_args);
        let output_return_def = crate::burn::codegen_return_expr(&self.graph_output_args);

        let mut body = quote! {};
        for (index, node) in self.nodes.iter().enumerate() {
            let mut scope_at_pos = self.scope.at_position(index);
            let code = NodeCodegen::forward(node, &mut scope_at_pos);
            body.extend(code);
        }

        // TODO Return the result without a `let` binding from a block,
        // otherwise let_and_return error will be triggered by clippy.
        // For now, we just disable the warning.
        quote! {
            #[allow(clippy::let_and_return, clippy::approx_constant)]
            pub fn forward(&self, #input_def) -> #output_type_def {
                #body

                #output_return_def
            }
        }
    }

    /// Register the input and output types of the graph using the passed in names.
    /// The names must be unique and match the names of the inputs and outputs of the nodes.
    /// The order will be preserved.
    ///
    /// # Arguments
    ///
    /// * `input_names` - The names of the inputs of the graph.
    /// * `output_names` - The names of the outputs of the graph.
    /// * `input_args` - The input arguments (from ONNX graph, used for empty graphs).
    /// * `output_args` - The output arguments (from ONNX graph, used for empty graphs).
    pub fn register_input_output(
        &mut self,
        input_names: Vec<String>,
        output_names: Vec<String>,
        input_args: &[onnx_ir::Argument],
        output_args: &[onnx_ir::Argument],
    ) {
        // Handle empty graphs: use provided arguments directly
        if self.nodes.is_empty() {
            // For empty graphs, inputs pass through directly to outputs
            self.graph_input_args.extend_from_slice(input_args);
            self.graph_output_args.extend_from_slice(output_args);
            return;
        }

        // Get the unique names of each input/output of the nodes
        let mut inputs = HashMap::new();
        let mut outputs = HashMap::new();
        for node in self.nodes.iter() {
            for input_arg in NodeCodegen::inputs(node) {
                inputs.insert(input_arg.name.clone(), input_arg.clone());
            }
            for output_arg in NodeCodegen::outputs(node) {
                outputs.insert(output_arg.name.clone(), output_arg.clone());
            }
        }

        // Get the input arguments of the graph using passed in names
        // For outer scope variables, fall back to the provided input_args
        input_names.iter().enumerate().for_each(|(idx, input)| {
            let input_arg = inputs
                .get(input)
                .cloned()
                .or_else(|| {
                    // Fall back to provided input_args for outer scope variables
                    if idx < input_args.len() {
                        Some(input_args[idx].clone())
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| panic!("Input argument not found for {input}"));

            self.graph_input_args.push(input_arg);
        });

        // Handle outputs - if output_args is provided (from ONNX), use it with renaming
        // Otherwise, look up arguments from node outputs (for tests)
        if !output_args.is_empty() {
            output_names
                .iter()
                .zip(output_args.iter())
                .for_each(|(name, arg)| {
                    // Rename argument to the graph output name
                    let mut renamed_arg = arg.clone();
                    renamed_arg.name = name.clone();
                    self.graph_output_args.push(renamed_arg);
                });
        } else {
            // For tests and non-ONNX usage: look up arguments from node outputs
            output_names.iter().for_each(|output| {
                self.graph_output_args.push(
                    outputs
                        .get(output)
                        .unwrap_or_else(|| panic!("Output argument not found for {output}"))
                        .clone(),
                );
            });
        }
    }
}
