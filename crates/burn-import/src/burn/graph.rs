use super::{BurnImports, Scope, Type};
use crate::burn::{
    TensorType,
    node::{Node, NodeCodegen},
};
use burn::record::{
    BinFileRecorder, BurnRecord, FileRecorder, NamedMpkFileRecorder, NamedMpkGzFileRecorder,
    PrecisionSettings, PrettyJsonFileRecorder, Recorder,
};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
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
    nodes: Vec<Node<PS>>,
    scope: Scope,
    imports: BurnImports,
    top_comment: Option<String>,
    default: Option<TokenStream>,
    blank_spaces: bool,
    graph_input_types: Vec<Type>,
    graph_output_types: Vec<Type>,
    _ps: PhantomData<PS>,
}

// The backend used for recording.
type Backend = burn_ndarray::NdArray;

impl<PS: PrecisionSettings> BurnGraph<PS> {
    /// Register a new operation node into the graph.
    ///
    /// # Notes
    ///
    /// The node must be registered in the same order they will be executed in the forward pass.
    pub fn register<N: NodeCodegen<PS> + 'static>(&mut self, node: N) {
        let node = node.into_node();
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
                        BurnGraphState::new(&self.nodes),
                    )),
                    out_file.clone(),
                )
                .unwrap();

                let mut layer_files: Vec<(syn::Ident, PathBuf)> = Vec::new();

                for node in &self.nodes {
                    let mut out_file = out_file.clone();
                    let node_type = match node.field_type() {
                        Some(node_type) => node_type.name().clone(),
                        None => continue,
                    };
                    out_file.set_file_name(node_type.to_string());

                    layer_files.push((node_type, out_file.clone()));
                    let _ = Recorder::<Backend>::save_item(
                        &recorder,
                        BurnRecord::<_, Backend>::new::<PrettyJsonFileRecorder<PS>>(node),
                        out_file,
                    );
                }

                assert!(
                    !embed_states,
                    "Embedding states is not supported for PrettyJsonFileRecorder."
                );

                self.register_record_file(
                    out_file,
                    &format!("burn::record::PrettyJsonFileRecorder::<{precision_ty_str}>"),
                    layer_files,
                );
            }
            RecordType::NamedMpkGz => {
                let recorder = NamedMpkGzFileRecorder::<PS>::new();

                Recorder::<Backend>::save_item(
                    &recorder,
                    BurnRecord::<_, Backend>::new::<NamedMpkGzFileRecorder<PS>>(StructMap(
                        BurnGraphState::new(&self.nodes),
                    )),
                    out_file.clone(),
                )
                .unwrap();

                let mut layer_files: Vec<(syn::Ident, PathBuf)> = Vec::new();

                for node in &self.nodes {
                    let mut out_file = out_file.clone();
                    let node_type = match node.field_type() {
                        Some(node_type) => node_type.name().clone(),
                        None => continue,
                    };
                    out_file.set_file_name(node_type.to_string());

                    layer_files.push((node_type, out_file.clone()));
                    let _ = Recorder::<Backend>::save_item(
                        &recorder,
                        BurnRecord::<_, Backend>::new::<NamedMpkGzFileRecorder<PS>>(node),
                        out_file,
                    );
                }

                assert!(
                    !embed_states,
                    "Embedding states is not supported for NamedMpkGzFileRecorder."
                );
                self.register_record_file(
                    out_file,
                    &format!("burn::record::NamedMpkGzFileRecorder::<{precision_ty_str}>"),
                    layer_files,
                );
            }

            RecordType::NamedMpk => {
                let recorder = NamedMpkFileRecorder::<PS>::new();

                Recorder::<Backend>::save_item(
                    &recorder,
                    BurnRecord::<_, Backend>::new::<NamedMpkGzFileRecorder<PS>>(StructMap(
                        BurnGraphState::new(&self.nodes),
                    )),
                    out_file.clone(),
                )
                .unwrap();

                let mut layer_files: Vec<(syn::Ident, PathBuf)> = Vec::new();

                for node in &self.nodes {
                    let mut out_file = out_file.clone();
                    let node_type = match node.field_type() {
                        Some(node_type) => node_type.name().clone(),
                        None => continue,
                    };
                    out_file.set_file_name(node_type.to_string());

                    layer_files.push((node_type, out_file.clone()));
                    let _ = Recorder::<Backend>::save_item(
                        &recorder,
                        BurnRecord::<_, Backend>::new::<NamedMpkGzFileRecorder<PS>>(node),
                        out_file,
                    );
                }

                assert!(
                    !embed_states,
                    "Embedding states is not supported for NamedMpkFileRecorder."
                );

                self.register_record_file(
                    out_file,
                    &format!("burn::record::NamedMpkFileRecorder::<{precision_ty_str}>"),
                    layer_files,
                );
            }

            RecordType::Bincode => {
                let recorder = BinFileRecorder::<PS>::new();

                let burn_graph_state = BurnGraphState::new(&self.nodes);
                let struct_tuple = StructTuple(burn_graph_state);

                Recorder::<Backend>::save_item(
                    &recorder,
                    BurnRecord::<_, Backend>::new::<BinFileRecorder<PS>>(struct_tuple),
                    out_file.clone(),
                )
                .unwrap();

                let mut layer_files: Vec<(syn::Ident, PathBuf)> = Vec::new();

                for node in &self.nodes {
                    let mut out_file = out_file.clone();
                    let node_type = match node.field_type() {
                        Some(node_type) => node_type.name().clone(),
                        None => continue,
                    };
                    out_file.set_file_name(node_type.to_string());

                    layer_files.push((node_type, out_file.clone()));
                    let _ = Recorder::<Backend>::save_item(
                        &recorder,
                        BurnRecord::<_, Backend>::new::<NamedMpkGzFileRecorder<PS>>(node),
                        out_file,
                    );
                }

                if embed_states {
                    self.register_record_embed(out_file, layer_files);
                } else {
                    self.register_record_file(
                        out_file,
                        &format!("burn::record::BinFileRecorder::<{precision_ty_str}>"),
                        layer_files,
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

        let forward_node_result: Vec<_> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(index, node)| node.forward(&mut self.scope, index))
            .collect();

        let codegen_imports = self.imports.codegen();
        let codegen_layer_loader_struct = self.codegen_layer_loader_struct();
        let codegen_layer_loader_new = self.codegen_layer_loader_new();
        let codegen_layer_loader_unloader = self.codegen_layer_loader_unloader();
        let codegen_layer_loader_loader = self.codegen_layer_loader_loader();
        let codegen_model_layer_forward =
            self.codegen_model_layer_forward(forward_node_result.clone());
        let codegen_struct = self.codegen_struct();
        let codegen_new = self.codegen_new();
        let codegen_forward = self.codegen_forward(forward_node_result);

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
            #maybe_top_file_comment
            #codegen_imports
            #maybe_blank
            #maybe_blank

            #codegen_layer_loader_struct
            #maybe_blank

            impl<B: Backend> ModelSegmentedLayerLoader<B> {
                #codegen_layer_loader_new
                #maybe_blank

                #codegen_layer_loader_unloader
                #maybe_blank

                #codegen_layer_loader_loader

                #codegen_model_layer_forward
            }

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
            .for_each(|node| node.register_imports(&mut self.imports));
    }
    /// Build the scope state to make sure tensor clones are added where needed.
    fn build_scope(&mut self) {
        log::debug!("Building the scope nodes len => '{}'", self.nodes.len());

        fn to_tensor(ty: Type) -> Option<TensorType> {
            match ty {
                Type::Tensor(tensor) => Some(tensor),
                Type::Scalar(_) => None,
                Type::Other(_) => None,
                Type::Shape(_) => None,
            }
        }

        // Register graph tensor input with 0 as node position
        self.graph_input_types
            .clone()
            .into_iter()
            .flat_map(to_tensor)
            .for_each(|tensor| {
                self.scope.tensor_register_variable(&tensor, 0);
            });

        self.nodes
            .iter()
            .enumerate()
            .for_each(|(node_position, node)| {
                node.output_types()
                    .into_iter()
                    .flat_map(to_tensor)
                    .for_each(|tensor| {
                        self.scope
                            .tensor_register_variable(&tensor, node_position + 1);
                    });
                // Since the graph is guaranteed to be a DAG, we can safely register future uses
                // of the inputs (which are the previous nodes' outputs)
                node.input_types()
                    .into_iter()
                    .flat_map(to_tensor)
                    .for_each(|tensor| {
                        self.scope
                            .tensor_register_future_use(&tensor, node_position)
                    });
            });

        // Register graph tensor output with the last node position
        self.graph_output_types
            .clone()
            .into_iter()
            .flat_map(to_tensor)
            .for_each(|tensor| {
                self.scope
                    .tensor_register_future_use(&tensor, self.nodes.len());
            });
    }

    fn register_record_file(
        &mut self,
        file: PathBuf,
        recorder_str: &str,
        layer_files: Vec<(syn::Ident, PathBuf)>,
    ) {
        self.imports.register("burn::record::Recorder");

        let recorder_ty = syn::parse_str::<syn::Type>(recorder_str).unwrap();

        let mut layer_states = quote! {};
        for (layer_file_name, layer_file_path) in layer_files {
            let layer_state_ident =
                format_ident!("{}_STATES", layer_file_name.to_string().to_uppercase());
            let layer_file_path = layer_file_path.to_str().unwrap();
            layer_states.extend(quote! {
                static #layer_state_ident: &str = #layer_file_path;
            });
        }

        // Add default implementation
        let file = file.to_str().unwrap();
        self.default = Some(quote! {
            _blank_!();
            #layer_states
            _blank_!();
            impl<B: Backend> ModelSegmentedLayerLoader<B> {
                pub fn recorder() -> #recorder_ty {
                    #recorder_ty::new()
                }
            }
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

    fn register_record_embed(&mut self, file: PathBuf, layer_files: Vec<(syn::Ident, PathBuf)>) {
        self.imports.register("burn::record::Recorder");

        // NOTE: Bincode format is used for embedding states for now.
        let precision = extract_type_name_by_type::<PS>();
        let precision_ty = syn::parse_str::<syn::Type>(&precision).unwrap();
        self.imports.register("burn::record::BinBytesRecorder");

        let mut layer_states = quote! {};
        for (layer_file_name, mut layer_file_path) in layer_files {
            let layer_state_ident =
                format_ident!("{}_STATES", layer_file_name.to_string().to_uppercase());
            layer_file_path
                .set_extension(<BinFileRecorder<PS> as FileRecorder<Backend>>::file_extension());
            let layer_file_path = layer_file_path.to_str().unwrap();
            layer_states.extend(quote! {
                static #layer_state_ident: &[u8] = include_bytes!(#layer_file_path);
            });
        }

        let mut file = file;
        file.set_extension(<BinFileRecorder<PS> as FileRecorder<Backend>>::file_extension());
        let file = file.to_str().unwrap();
        self.default = Some(quote! {
            _blank_!();
            #layer_states
            _blank_!();
            impl<B: Backend> ModelSegmentedLayerLoader<B> {
                pub fn recorder() -> BinBytesRecorder<#precision_ty, &'static [u8]> {
                    BinBytesRecorder::default()
                }
            }
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
                    .load(EMBEDDED_STATES.into(), device)
                    .expect("Should decode state successfully");

                    Self::new(device).load_record(record)
                }
            }

        });
    }

    fn codegen_layer_loader_struct(&self) -> TokenStream {
        let mut body = quote! {};
        self.nodes
            .iter()
            .filter_map(|node| node.field_type())
            .map(|field| {
                let name = field.name();
                let ty = field.ty();

                if matches!(&field, Type::Tensor(_)) {
                    quote! {
                        #name: Option<burn::module::Param<#ty>>,
                    }
                } else {
                    quote! {
                        #name: Option<#ty>,
                    }
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
            pub struct ModelSegmentedLayerLoader<B: Backend> {
                #body
            }
        }
    }

    fn codegen_struct(&self) -> TokenStream {
        let mut body = quote! {};
        self.nodes
            .iter()
            .filter_map(|node| node.field_type())
            .map(|field| {
                let name = field.name();
                let ty = field.ty();

                if matches!(&field, Type::Tensor(_)) {
                    quote! {
                        #name: burn::module::Param<#ty>,
                    }
                } else {
                    quote! {
                        #name: #ty,
                    }
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

    fn codegen_layer_loader_loader(&self) -> TokenStream {
        let fields = self
            .nodes
            .iter()
            .filter_map(|node| {
                let field_type = node.field_type()?;
                let field_init = node.field_init()?;
                Some((
                    field_type.name().clone(),
                    field_type.ty().clone(),
                    field_init.clone(),
                ))
            })
            .collect::<Vec<_>>();

        let mut body = quote! {};
        for (field_name, field_ty, field_init) in fields {
            let field_ty_parsed: syn::Path =
                syn::parse2::<syn::Path>(field_ty).expect("Failed to parse field type");
            let field_ty_parsed = &field_ty_parsed
                .segments
                .last()
                .expect("Their should be some sort of segment to define the type");
            let field_ty_ident = &field_ty_parsed.ident;

            let function_name = format_ident!("load_{}", field_name);
            let record_name = format_ident!("{}Record", quote! {#field_ty_ident}.to_string());
            let state_name = format_ident!("{}_STATES", field_name.to_string().to_uppercase());

            body.extend(quote! {
                #[allow(unused)]
                pub fn #function_name(&mut self, device: &B::Device) {
                    let record: #record_name<B> = Self::recorder()
                        .load(#state_name.into(), device)
                        .expect("Should decode state successfully");

                    #field_init
                    self.#field_name = Some(burn::module::Module::<B>::load_record(#field_name, record));
                }
            });
        }

        body
    }

    fn codegen_layer_loader_unloader(&self) -> TokenStream {
        let fields = self
            .nodes
            .iter()
            .flat_map(|node| node.field_type())
            .map(|field| field.name().clone())
            .collect::<Vec<_>>();

        let mut body = quote! {};
        for field_name in fields {
            let function_name = format_ident!("unload_{}", field_name);
            body.extend(quote! {
                #[allow(unused)]
                pub fn #function_name(&mut self) {
                    self.#field_name = None;
                }
            });
        }

        body
    }

    fn codegen_layer_loader_new(&self) -> TokenStream {
        let fields = self
            .nodes
            .iter()
            .flat_map(|node| node.field_type())
            .map(|field| {
                let name = field.name().clone();
                quote! { #name: None }
            })
            .collect::<Vec<_>>();

        quote! {
            #[allow(unused_variables)]
            pub fn new(device: &B::Device) -> Self {
                Self {
                    #(#fields,)*
                    phantom: core::marker::PhantomData,
                    device: burn::module::Ignored(device.clone()),
                }
            }
        }
    }

    fn codegen_new(&self) -> TokenStream {
        let mut body = quote! {};

        self.nodes
            .iter()
            .map(|node| node.field_init())
            .for_each(|code| body.extend(code));

        let fields = self
            .nodes
            .iter()
            .flat_map(|node| node.field_type())
            .map(|field| field.name().clone())
            .collect::<Vec<_>>();

        quote! {
            #[allow(unused_variables)]
            pub fn new(device: &B::Device) -> Self {
                #body

                Self {
                    #(#fields,)*
                    phantom: core::marker::PhantomData,
                    device: burn::module::Ignored(device.clone()),
                }
            }
        }
    }

    fn codegen_model_layer_forward(&mut self, forward_functions: Vec<TokenStream>) -> TokenStream {
        let mut input_def = quote! {};
        let mut output_type_def = quote! {};
        let mut output_return_def = quote! {};

        self.graph_input_types.iter().for_each(|input| {
            let name = input.name().clone();
            let ty = input.ty();

            input_def.extend(quote! {
                #name: #ty,

            })
        });

        input_def.extend(quote! {
            device: &B::Device,
        });

        let multiple_output = self.graph_output_types.len() > 1;

        self.graph_output_types.iter().for_each(|output| {
            let name = output.name();
            let ty = output.ty();

            if multiple_output {
                output_type_def.extend(quote! {
                    #ty,
                });
                output_return_def.extend(quote! {
                    #name,
                });
            } else {
                output_type_def.extend(quote! {
                    #ty
                });
                output_return_def.extend(quote! {
                    #name
                });
            }
        });

        if multiple_output {
            output_return_def = quote! {
                (#output_return_def)
            };
            output_type_def = quote! {
                (#output_type_def)
            };
        }

        let mut body = quote! {};
        self.nodes
            .iter()
            .zip(forward_functions)
            .for_each(|(node, code)| {
                if let Some(field_type) = node.field_type() {
                    let name = field_type.name();

                    let code_str = code.to_string();
                    let replaced = code_str.replace(
                        &format!("self . {}", name),
                        &format!("self . {} . take () ?", name),
                    );
                    let code: TokenStream = replaced.parse().unwrap();

                    let load_function_name = format_ident!("load_{}", name);
                    body.extend(quote! {self.#load_function_name(device);});
                    body.extend(code);
                } else {
                    body.extend(code);
                }
            });

        // TODO Return the result without a `let` binding from a block,
        // otherwise let_and_return error will be triggered by clippy.
        // For now, we just disable the warning.
        quote! {
            #[allow(clippy::let_and_return, clippy::approx_constant)]
            pub fn memory_efficient_forward(&mut self, #input_def) -> Option<#output_type_def> {
                #body
                Some(#output_return_def)
            }
        }
    }

    fn codegen_forward(&mut self, forward_functions: Vec<TokenStream>) -> TokenStream {
        let mut input_def = quote! {};
        let mut output_type_def = quote! {};
        let mut output_return_def = quote! {};

        self.graph_input_types.iter().for_each(|input| {
            let name = input.name().clone();
            let ty = input.ty();

            input_def.extend(quote! {
                #name: #ty,

            })
        });

        let multiple_output = self.graph_output_types.len() > 1;

        self.graph_output_types.iter().for_each(|output| {
            let name = output.name();
            let ty = output.ty();

            if multiple_output {
                output_type_def.extend(quote! {
                    #ty,
                });
                output_return_def.extend(quote! {
                    #name,
                });
            } else {
                output_type_def.extend(quote! {
                    #ty
                });
                output_return_def.extend(quote! {
                    #name
                });
            }
        });

        if multiple_output {
            output_return_def = quote! {
                (#output_return_def)
            };
            output_type_def = quote! {
                (#output_type_def)
            };
        }

        let mut body = quote! {};
        forward_functions
            .into_iter()
            .for_each(|code| body.extend(code));

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
    ///
    /// # Panics
    ///
    /// Panics if the graph is empty.
    pub fn register_input_output(&mut self, input_names: Vec<String>, output_names: Vec<String>) {
        assert!(
            !self.nodes.is_empty(),
            "Cannot register input and output types for an empty graph."
        );

        // Get the unique names of each input of the nodes
        let mut inputs = HashMap::new();
        let mut outputs = HashMap::new();
        for node in self.nodes.iter() {
            for input in node.input_types() {
                inputs.insert(input.name().to_string(), input);
            }
            for output in node.output_types() {
                outputs.insert(output.name().to_string(), output);
            }
        }

        // Get the input and output types of the graph using passed in names
        input_names.iter().for_each(|input| {
            self.graph_input_types.push(
                inputs
                    .get(&Type::format_name(input))
                    .unwrap_or_else(|| panic!("Input type not found for {input}"))
                    .clone(),
            );
        });

        output_names.iter().for_each(|output| {
            self.graph_output_types.push(
                outputs
                    .get(&Type::format_name(output))
                    .unwrap_or_else(|| panic!("Output type not found for {output}"))
                    .clone(),
            );
        });
    }
}

#[derive(new, Debug)]
struct BurnGraphState<'a, PS: PrecisionSettings> {
    nodes: &'a Vec<Node<PS>>,
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

impl<PS: PrecisionSettings> Serialize for StructMap<'_, PS> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let nodes_with_names = self
            .0
            .nodes
            .iter()
            .filter_map(|node| node.field_type().map(|ty| (node, ty.name().clone())))
            .collect::<Vec<_>>();
        let mut map = serializer.serialize_map(Some(nodes_with_names.len()))?;

        for (node, name) in nodes_with_names.iter() {
            map.serialize_entry(&name.to_string(), &node)?;
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

impl<PS: PrecisionSettings> Serialize for StructTuple<'_, PS> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let nodes_with_names = self
            .0
            .nodes
            .iter()
            .filter_map(|node| node.field_type().map(|ty| (node, ty.name().clone())))
            .collect::<Vec<_>>();
        let mut map = serializer.serialize_tuple(nodes_with_names.len())?;

        for (node, _name) in nodes_with_names.iter() {
            map.serialize_element(&node)?;
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
