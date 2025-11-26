use super::{BurnImports, Scope};
use crate::burn::node::NodeCodegen;
use burn::record::{
    BinFileRecorder, BurnRecord, FileRecorder, NamedMpkFileRecorder, NamedMpkGzFileRecorder,
    PrecisionSettings, PrettyJsonFileRecorder, Recorder,
};
use onnx_ir::{Node, ir::ArgType};
use proc_macro2::TokenStream;
use quote::quote;
use serde::{
    Serialize,
    ser::{SerializeMap, SerializeTuple},
};
use std::{any::type_name, collections::HashMap, marker::PhantomData, path::PathBuf};

/// Type of the record to be saved.
#[derive(Debug, Clone, Default, Copy)]
pub enum RecordType {
    /// Pretty JSON format (useful for debugging).
    PrettyJson,

    /// Compressed Named MessagePack.
    ///
    /// Note: This may cause infinite build.
    ///       See [#952 bug](https://github.com/tracel-ai/burn/issues/952).
    NamedMpkGz,

    /// Uncompressed Named MessagePack.
    #[default]
    NamedMpk,

    /// Bincode format (useful for embedding and for no-std support).
    Bincode,
}

/// Burn graph intermediate representation of modules and tensor operations.
#[derive(Default, Debug)]
pub struct BurnGraph<PS: PrecisionSettings> {
    nodes: Vec<Node>,
    scope: Scope,
    imports: BurnImports,
    top_comment: Option<String>,
    default: Option<TokenStream>,
    blank_spaces: bool,
    graph_input_args: Vec<onnx_ir::Argument>,
    graph_output_args: Vec<onnx_ir::Argument>,
    _ps: PhantomData<PS>,
}

// The backend used for recording.
type Backend = burn_ndarray::NdArray;

impl<PS: PrecisionSettings + 'static> BurnGraph<PS> {
    /// Register a new operation node into the graph.
    ///
    /// # Notes
    ///
    /// The node must be registered in the same order they will be executed in the forward pass.
    pub fn register(&mut self, node: Node) {
        log::debug!("Registering node => '{}'", node.name());
        self.nodes.push(node);
    }

    /// Save the state of each node in a record file.
    ///
    /// The `Default` trait will be implemented for the generated model, which will load the record
    /// saved at the provided path. In case of `embed_states` is true, the record will be embedded
    /// in the generated code (useful for no-std support).
    ///
    /// # Arguments
    ///
    /// * `out_file` - The path to the record file.
    /// * `record_type` - The type of the record to be saved.
    /// * `embed_states` - Embed the record in the generated code.
    ///
    /// # Panics
    ///
    /// Panics if the record type is not `RecordType::Bincode` and `embed_states` is `true`.
    pub fn with_record(
        mut self,
        out_file: PathBuf,
        record_type: RecordType,
        embed_states: bool,
    ) -> Self {
        let precision_ty_str = extract_type_name_by_type::<PS>();
        self.imports
            .register(format!("burn::record::{precision_ty_str}"));

        match record_type {
            RecordType::PrettyJson => {
                let recorder = PrettyJsonFileRecorder::<PS>::new();

                Recorder::<Backend>::save_item(
                    &recorder,
                    BurnRecord::<_, Backend>::new::<PrettyJsonFileRecorder<PS>>(StructMap(
                        BurnGraphState::<PS>::new(&self.nodes),
                    )),
                    out_file.clone(),
                )
                .unwrap();

                assert!(
                    !embed_states,
                    "Embedding states is not supported for PrettyJsonFileRecorder."
                );

                self.register_record_file(
                    out_file,
                    &format!("burn::record::PrettyJsonFileRecorder::<{precision_ty_str}>"),
                );
            }
            RecordType::NamedMpkGz => {
                let recorder = NamedMpkGzFileRecorder::<PS>::new();

                Recorder::<Backend>::save_item(
                    &recorder,
                    BurnRecord::<_, Backend>::new::<NamedMpkGzFileRecorder<PS>>(StructMap(
                        BurnGraphState::<PS>::new(&self.nodes),
                    )),
                    out_file.clone(),
                )
                .unwrap();

                assert!(
                    !embed_states,
                    "Embedding states is not supported for NamedMpkGzFileRecorder."
                );
                self.register_record_file(
                    out_file,
                    &format!("burn::record::NamedMpkGzFileRecorder::<{precision_ty_str}>"),
                );
            }

            RecordType::NamedMpk => {
                let recorder = NamedMpkFileRecorder::<PS>::new();

                Recorder::<Backend>::save_item(
                    &recorder,
                    BurnRecord::<_, Backend>::new::<NamedMpkFileRecorder<PS>>(StructMap(
                        BurnGraphState::<PS>::new(&self.nodes),
                    )),
                    out_file.clone(),
                )
                .unwrap();

                assert!(
                    !embed_states,
                    "Embedding states is not supported for NamedMpkFileRecorder."
                );

                self.register_record_file(
                    out_file,
                    &format!("burn::record::NamedMpkFileRecorder::<{precision_ty_str}>"),
                );
            }

            RecordType::Bincode => {
                let recorder = BinFileRecorder::<PS>::new();

                Recorder::<Backend>::save_item(
                    &recorder,
                    BurnRecord::<_, Backend>::new::<BinFileRecorder<PS>>(StructTuple(
                        BurnGraphState::<PS>::new(&self.nodes),
                    )),
                    out_file.clone(),
                )
                .unwrap();

                if embed_states {
                    self.register_record_embed(out_file);
                } else {
                    self.register_record_file(
                        out_file,
                        &format!("burn::record::BinFileRecorder::<{precision_ty_str}>"),
                    );
                }
            }
        }

        self
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
    }

    fn register_imports(&mut self) {
        // Register imports from nodes
        self.nodes
            .iter()
            .for_each(|node| <Node as NodeCodegen<PS>>::register_imports(node, &mut self.imports));
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

    fn register_record_file(&mut self, file: PathBuf, recorder_str: &str) {
        self.imports.register("burn::record::Recorder");

        let recorder_ty = syn::parse_str::<syn::Type>(recorder_str).unwrap();

        // Add default implementation
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
                pub fn from_file(file: &str, device: &B::Device) -> Self {
                    let record = #recorder_ty::new()
                        .load(file.into(), device)
                        .expect("Record file to exist.");
                    Self::new(device).load_record(record)
                }
            }
        });
    }

    fn register_record_embed(&mut self, file: PathBuf) {
        self.imports.register("burn::record::Recorder");

        // NOTE: Bincode format is used for embedding states for now.
        let precision = extract_type_name_by_type::<PS>();
        let precision_ty = syn::parse_str::<syn::Type>(&precision).unwrap();
        self.imports.register("burn::record::BinBytesRecorder");

        let mut file = file;
        file.set_extension(<BinFileRecorder<PS> as FileRecorder<Backend>>::file_extension());
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
                pub fn from_embedded(device: &B::Device) -> Self {
                    let record = BinBytesRecorder::<#precision_ty, &'static [u8]>::default()
                    .load(EMBEDDED_STATES, device)
                    .expect("Should decode state successfully");

                    Self::new(device).load_record(record)
                }
            }

        });
    }

    /// Recursively collect all fields from nodes, including subgraph nodes in If/Loop/Scan
    fn collect_all_fields(&self) -> Vec<(proc_macro2::Ident, TokenStream, Option<TokenStream>)> {
        use std::collections::HashMap;

        // Track field name usage to make them unique
        let mut field_name_counts: HashMap<String, usize> = HashMap::new();
        let mut all_fields: Vec<(proc_macro2::Ident, TokenStream, Option<TokenStream>)> =
            Vec::new();

        // Helper to recursively collect fields from a subgraph and its nested subgraphs
        // Used by both If and Loop nodes
        fn collect_subgraph_fields_recursive<PS: PrecisionSettings + 'static>(
            subgraph: &onnx_ir::OnnxGraph,
            field_name_counts: &mut HashMap<String, usize>,
            all_fields: &mut Vec<(proc_macro2::Ident, TokenStream, Option<TokenStream>)>,
        ) {
            for onnx_node in &subgraph.nodes {
                let burn_node = onnx_node;
                // Collect this node's field if it has one
                if let Some(mut field) = NodeCodegen::<PS>::field(burn_node) {
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
                    collect_subgraph_fields_recursive::<PS>(
                        &nested_if_node.config.then_branch,
                        field_name_counts,
                        all_fields,
                    );
                    collect_subgraph_fields_recursive::<PS>(
                        &nested_if_node.config.else_branch,
                        field_name_counts,
                        all_fields,
                    );
                } else if let Node::Loop(nested_loop_node) = burn_node {
                    collect_subgraph_fields_recursive::<PS>(
                        &nested_loop_node.config.body,
                        field_name_counts,
                        all_fields,
                    );
                }
            }
        }

        for node in &self.nodes {
            // Collect this node's field if it has one
            if let Some(field) = NodeCodegen::<PS>::field(node) {
                all_fields.push((field.name, field.ty, Some(field.init)));
            }

            // Recursively collect fields from If/Loop node subgraphs
            // Note: Subgraph fields are NOT deduplicated - each branch has unique fields
            if let Node::If(if_node) = node {
                collect_subgraph_fields_recursive::<PS>(
                    &if_node.config.then_branch,
                    &mut field_name_counts,
                    &mut all_fields,
                );
                collect_subgraph_fields_recursive::<PS>(
                    &if_node.config.else_branch,
                    &mut field_name_counts,
                    &mut all_fields,
                );
            } else if let Node::Loop(loop_node) = node {
                collect_subgraph_fields_recursive::<PS>(
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
            let code = <Node as NodeCodegen<PS>>::forward(node, &mut scope_at_pos);
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
            for input_arg in <Node as NodeCodegen<PS>>::inputs(node) {
                inputs.insert(input_arg.name.clone(), input_arg.clone());
            }
            for output_arg in <Node as NodeCodegen<PS>>::outputs(node) {
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

#[derive(new, Debug)]
struct BurnGraphState<'a, PS: PrecisionSettings> {
    nodes: &'a Vec<Node>,
    #[new(default)]
    _ps: PhantomData<PS>,
}

/// Represents a custom serialization strategy for the graph state in the module struct.
///
/// This struct serializes the graph state using a map format. Specifically, nodes are
/// serialized as a map where each node name acts as the key and the node itself is the value.
///
/// Notably, this approach is utilized by serialization formats such as PrettyJson, NamedMpk,
/// and NamedMpkGz.
///
/// # Notes
///
/// Mpk and Bincode cannot use this method because they do not support serializing maps.
/// Instead, they use the `StructTuple` serialization strategy (to avoid memory overhead presumably).
struct StructMap<'a, PS: PrecisionSettings>(BurnGraphState<'a, PS>);

impl<PS: PrecisionSettings + 'static> Serialize for StructMap<'_, PS> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use std::collections::HashMap;

        // Track field name usage to make them unique (same logic as collect_all_fields)
        let mut field_name_counts: HashMap<String, usize> = HashMap::new();
        let mut all_nodes: Vec<(String, Node)> = Vec::new();

        // Helper to recursively collect nodes from subgraphs
        // Used by both If and Loop nodes
        fn collect_subgraph_nodes_recursive<PS: PrecisionSettings + 'static>(
            subgraph: &onnx_ir::OnnxGraph,
            field_name_counts: &mut HashMap<String, usize>,
            all_nodes: &mut Vec<(String, Node)>,
        ) {
            for onnx_node in &subgraph.nodes {
                let burn_node = onnx_node;
                if let Some(field_type) = NodeCodegen::<PS>::field(burn_node) {
                    let base_name = field_type.name.to_string();
                    let count = field_name_counts.entry(base_name.clone()).or_insert(0);
                    *count += 1;

                    // Create unique name if needed
                    let unique_name = if *count > 1 {
                        format!("{}_{}", base_name, count)
                    } else {
                        base_name
                    };

                    all_nodes.push((unique_name, burn_node.clone()));
                }

                // Recursively collect from nested If/Loop nodes
                if let Node::If(nested_if_node) = burn_node {
                    collect_subgraph_nodes_recursive::<PS>(
                        &nested_if_node.config.then_branch,
                        field_name_counts,
                        all_nodes,
                    );
                    collect_subgraph_nodes_recursive::<PS>(
                        &nested_if_node.config.else_branch,
                        field_name_counts,
                        all_nodes,
                    );
                } else if let Node::Loop(nested_loop_node) = burn_node {
                    collect_subgraph_nodes_recursive::<PS>(
                        &nested_loop_node.config.body,
                        field_name_counts,
                        all_nodes,
                    );
                }
            }
        }

        // Add main graph nodes
        for node in self.0.nodes.iter() {
            if let Some(field_type) = NodeCodegen::<PS>::field(node) {
                let field_name = field_type.name.to_string();
                all_nodes.push((field_name, node.clone()));
            }

            // Add subgraph nodes from If/Loop nodes with unique names
            if let Node::If(if_node) = node {
                collect_subgraph_nodes_recursive::<PS>(
                    &if_node.config.then_branch,
                    &mut field_name_counts,
                    &mut all_nodes,
                );
                collect_subgraph_nodes_recursive::<PS>(
                    &if_node.config.else_branch,
                    &mut field_name_counts,
                    &mut all_nodes,
                );
            } else if let Node::Loop(loop_node) = node {
                collect_subgraph_nodes_recursive::<PS>(
                    &loop_node.config.body,
                    &mut field_name_counts,
                    &mut all_nodes,
                );
            }
        }

        let mut map = serializer.serialize_map(Some(all_nodes.len()))?;

        for (name, node) in all_nodes.iter() {
            // Serialize the node's field using a wrapper
            struct NodeFieldSerializer<'a, PS: PrecisionSettings>(&'a Node, PhantomData<PS>);

            impl<PS: PrecisionSettings + 'static> Serialize for NodeFieldSerializer<'_, PS> {
                fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
                where
                    S: serde::Serializer,
                {
                    <Node as NodeCodegen<PS>>::field_serialize(self.0, serializer)
                }
            }

            map.serialize_entry(
                &name.to_string(),
                &NodeFieldSerializer::<PS>(node, PhantomData),
            )?;
        }

        map.end()
    }
}

/// Represents a custom serialization strategy for the graph state in the module struct.
///
/// In contrast to `StructMap`, this struct serializes the graph state using a tuple format.
/// Each node is simply serialized as an element of the tuple without explicit naming.
///
/// Serialization formats such as Mpk and Bincode employ this method.
///
/// # Notes
///
/// PrettyJson, NamedMpk, and NamedMpkGz cannot use this method because they do not support
/// serializing tuples. Instead, they use the `StructMap` serialization strategy.
struct StructTuple<'a, PS: PrecisionSettings>(BurnGraphState<'a, PS>);

impl<PS: PrecisionSettings + 'static> Serialize for StructTuple<'_, PS> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use std::collections::HashMap;

        // Track field name usage (same as other methods, for consistency)
        let mut field_name_counts: HashMap<String, usize> = HashMap::new();
        let mut all_nodes: Vec<Node> = Vec::new();

        // Helper to recursively collect nodes from subgraphs
        // Used by both If and Loop nodes
        fn collect_subgraph_nodes_recursive<PS: PrecisionSettings + 'static>(
            subgraph: &onnx_ir::OnnxGraph,
            field_name_counts: &mut HashMap<String, usize>,
            all_nodes: &mut Vec<Node>,
        ) {
            for onnx_node in &subgraph.nodes {
                let burn_node = onnx_node;
                if let Some(field_type) = NodeCodegen::<PS>::field(burn_node) {
                    let base_name = field_type.name.to_string();
                    let count = field_name_counts.entry(base_name.clone()).or_insert(0);
                    *count += 1;

                    // Just track the count for ordering consistency
                    all_nodes.push(burn_node.clone());
                }

                // Recursively collect from nested If/Loop nodes
                if let Node::If(nested_if_node) = burn_node {
                    collect_subgraph_nodes_recursive::<PS>(
                        &nested_if_node.config.then_branch,
                        field_name_counts,
                        all_nodes,
                    );
                    collect_subgraph_nodes_recursive::<PS>(
                        &nested_if_node.config.else_branch,
                        field_name_counts,
                        all_nodes,
                    );
                } else if let Node::Loop(nested_loop_node) = burn_node {
                    collect_subgraph_nodes_recursive::<PS>(
                        &nested_loop_node.config.body,
                        field_name_counts,
                        all_nodes,
                    );
                }
            }
        }

        // Add main graph nodes
        for node in self.0.nodes.iter() {
            if NodeCodegen::<PS>::field(node).is_some() {
                all_nodes.push(node.clone());
            }

            // Add subgraph nodes from If/Loop nodes
            // Apply same uniqueness logic even though tuple serialization doesn't use names
            if let Node::If(if_node) = node {
                collect_subgraph_nodes_recursive::<PS>(
                    &if_node.config.then_branch,
                    &mut field_name_counts,
                    &mut all_nodes,
                );
                collect_subgraph_nodes_recursive::<PS>(
                    &if_node.config.else_branch,
                    &mut field_name_counts,
                    &mut all_nodes,
                );
            } else if let Node::Loop(loop_node) = node {
                collect_subgraph_nodes_recursive::<PS>(
                    &loop_node.config.body,
                    &mut field_name_counts,
                    &mut all_nodes,
                );
            }
        }

        let mut map = serializer.serialize_tuple(all_nodes.len())?;

        for node in all_nodes.iter() {
            // Serialize the node's field using a wrapper
            struct NodeFieldSerializer<'a, PS: PrecisionSettings>(&'a Node, PhantomData<PS>);

            impl<PS: PrecisionSettings + 'static> Serialize for NodeFieldSerializer<'_, PS> {
                fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
                where
                    S: serde::Serializer,
                {
                    <Node as NodeCodegen<PS>>::field_serialize(self.0, serializer)
                }
            }

            map.serialize_element(&NodeFieldSerializer::<PS>(node, PhantomData))?;
        }

        map.end()
    }
}

fn extract_type_name_by_type<T: ?Sized>() -> String {
    let full_type_name = type_name::<T>();
    full_type_name
        .rsplit("::")
        .next()
        .unwrap_or(full_type_name)
        .to_string()
}
