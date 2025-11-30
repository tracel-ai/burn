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
                        BurnGraphState::new(&self.nodes),
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
                    BurnRecord::<_, Backend>::new::<NamedMpkGzFileRecorder<PS>>(StructMap(
                        BurnGraphState::new(&self.nodes),
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
                        BurnGraphState::new(&self.nodes),
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

    fn codegen_forward(&mut self) -> TokenStream {
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
        self.nodes
            .iter()
            .enumerate()
            .map(|(index, node)| node.forward(&mut self.scope, index))
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
    /// * `input_types` - The types of the inputs (from ONNX graph, used for empty graphs).
    /// * `output_types` - The types of the outputs (from ONNX graph, used for empty graphs).
    pub fn register_input_output(
        &mut self,
        input_names: Vec<String>,
        output_names: Vec<String>,
        input_types: &[Type],
        output_types: &[Type],
    ) {
        // Handle empty graphs: use provided types directly
        if self.nodes.is_empty() {
            // For empty graphs, inputs pass through directly to outputs
            self.graph_input_types.extend_from_slice(input_types);
            self.graph_output_types.extend_from_slice(output_types);
            return;
        }

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
                    .get(input)
                    .unwrap_or_else(|| panic!("Input type not found for {input}"))
                    .clone(),
            );
        });

        // Handle outputs - if output_types is provided (from ONNX), use it with renaming
        // Otherwise, look up types from node outputs (for tests)
        if !output_types.is_empty() {
            output_names
                .iter()
                .zip(output_types.iter())
                .for_each(|(name, ty)| {
                    // Use the type from onnx-ir but rename it to the graph output name
                    // (onnx-ir provides the resolved node output, we want the graph output name)
                    self.graph_output_types.push(ty.with_name(name));
                });
        } else {
            // For tests and non-ONNX usage: look up output types from node outputs
            output_names.iter().for_each(|output| {
                self.graph_output_types.push(
                    outputs
                        .get(output)
                        .unwrap_or_else(|| panic!("Output type not found for {output}"))
                        .clone(),
                );
            });
        }
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
