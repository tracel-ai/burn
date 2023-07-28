use super::{BurnImports, Scope, Type};
use crate::burn::{
    node::{Node, NodeCodegen},
    TensorType,
};
use burn::record::{
    BurnRecord, DefaultFileRecorder, FileRecorder, PrecisionSettings, PrettyJsonFileRecorder,
};
use proc_macro2::TokenStream;
use quote::quote;
use serde::{ser::SerializeMap, Serialize};
use std::path::PathBuf;

/// Burn graph intermediate representation of modules and tensor operations.
#[derive(Default, Debug)]
pub struct BurnGraph<PS: PrecisionSettings> {
    nodes: Vec<Node<PS>>,
    scope: Scope,
    imports: BurnImports,
    top_comment: Option<String>,
    default: Option<TokenStream>,
    blank_spaces: bool,
    gen_new_fn: bool,
}

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

    /// Generate a function `Model::new()` without any argument when `gen_new_fn` is `true`.
    ///
    /// This is useful if you intend to train the model generated.
    pub fn with_new_fn(mut self, gen_new_fn: bool) -> Self {
        self.gen_new_fn = gen_new_fn;
        self
    }

    /// Save the state of each node in a record file.
    ///
    /// The `Default` trait will be implemented for the generated model, which will load the record
    /// saved at the provided path.
    ///
    /// # Notes
    ///
    /// The development argument will change the recorder used.
    /// [pretty json](PrettyJsonFileRecorder) is used when development is true and [default](DefaultFileRecorder) is used otherwise.
    ///
    /// The precision type must be passed as `&str` and should be the same type definition as the
    /// `PS` graph generic argument. [type_name](std::any::type_name) can't be used reliably for
    /// that purpose.
    pub fn with_record(
        mut self,
        out_file: PathBuf,
        development: bool,
        precision_ty_str: &str,
    ) -> Self {
        if development {
            let recorder = PrettyJsonFileRecorder::<PS>::new();
            self.register_record(
                recorder,
                out_file,
                &format!("burn::record::PrettyJsonFileRecorder::<{precision_ty_str}>"),
            );
        } else {
            let recorder = DefaultFileRecorder::<PS>::new();
            self.register_record(
                recorder,
                out_file,
                &format!("burn::record::DefaultFileRecorder::<{precision_ty_str}>"),
            );
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

    /// Generate tokens reprensenting the graph with Burn modules and tensor operations.
    pub fn codegen(mut self) -> TokenStream {
        self.build_scope();
        self.nodes
            .iter()
            .for_each(|node| node.register_imports(&mut self.imports));

        let codegen_imports = self.imports.codegen();
        let codegen_struct = self.codegen_struct();
        let codegen_new_record = self.codegen_new_record();
        let codegen_forward = self.codegen_forward();

        let maybe_blank = match self.blank_spaces {
            true => quote! {
                _blank_!();
            },
            false => quote! {},
        };
        let codegen_new = match self.gen_new_fn {
            true => {
                let new_fn = self.codegen_new();
                quote! {
                    #new_fn
                    #maybe_blank
                }
            }
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

            #codegen_struct
            #maybe_blank

            #codegen_default

            impl<B: Backend> Model<B> {
                #codegen_new_record
                #maybe_blank

                #codegen_new
                #codegen_forward
            }
        }
    }
    /// Build the scope state to make sure tensor clones are added where needed.
    fn build_scope(&mut self) {
        log::debug!("Building the scope nodes len => '{}'", self.nodes.len());

        let input = self.nodes.first().unwrap();

        fn to_tensor(ty: Type<'_>) -> Option<&TensorType> {
            match ty {
                Type::Tensor(tensor) => Some(tensor),
                Type::Other(_) => None,
            }
        }

        input
            .input_types()
            .into_iter()
            .flat_map(to_tensor)
            .for_each(|tensor| self.scope.tensor_register_variable(tensor, 0));

        self.nodes
            .iter()
            .enumerate()
            .for_each(|(node_position, node)| {
                node.output_types()
                    .into_iter()
                    .flat_map(to_tensor)
                    .for_each(|tensor| {
                        self.scope
                            .tensor_register_variable(tensor, node_position + 1)
                    })
            });

        self.nodes
            .iter()
            .enumerate()
            .for_each(|(node_position, node)| {
                node.input_types()
                    .into_iter()
                    .flat_map(to_tensor)
                    .for_each(|tensor| self.scope.tensor_register_future_use(tensor, node_position))
            });
    }

    fn register_record<FR: FileRecorder>(
        &mut self,
        recorder: FR,
        file: PathBuf,
        recorder_str: &str,
    ) {
        self.imports.register("burn::record::Recorder");

        let state = BurnGraphState::new(&self.nodes);
        recorder
            .save_item(BurnRecord::new::<FR>(state), file.clone())
            .unwrap();

        let recorder_ty = syn::parse_str::<syn::Type>(recorder_str).unwrap();
        let file = file.to_str();

        // Add default implementation
        self.default = Some(quote! {
            impl<B: Backend> Default for Model<B> {
                fn default() -> Self {
                    let record = #recorder_ty::new()
                        .load(#file.into())
                        .expect("Record file to exist.");
                    Self::new_with(record)
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

                quote! {
                    #name: #ty,
                }
            })
            .for_each(|code| body.extend(code));

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
            .map(|node| node.field_init(false))
            .for_each(|code| body.extend(code));

        let fields = self
            .nodes
            .iter()
            .flat_map(|node| node.field_type())
            .map(|field| field.name().clone())
            .collect::<Vec<_>>();

        quote! {
            #[allow(dead_code)]
            pub fn new() -> Self {
                #body

                Self {
                    #(#fields,)*
                }
            }
        }
    }
    fn codegen_new_record(&self) -> TokenStream {
        let mut body = quote! {};

        self.nodes
            .iter()
            .map(|node| node.field_init(true))
            .for_each(|code| body.extend(code));

        let fields = self
            .nodes
            .iter()
            .flat_map(|node| node.field_type())
            .map(|field| field.name().clone())
            .collect::<Vec<_>>();

        quote! {
            pub fn new_with(record: ModelRecord<B>) -> Self {
                #body

                Self {
                    #(#fields,)*
                }
            }
        }
    }

    fn codegen_forward(&mut self) -> TokenStream {
        let mut input_def = quote! {};
        let mut output_type_def = quote! {};
        let mut output_return_def = quote! {};

        self.nodes
            .first()
            .unwrap()
            .input_types()
            .into_iter()
            .for_each(|input| {
                let name = input.name();
                let ty = input.ty();

                input_def.extend(quote! {
                    #name: #ty,

                })
            });

        let output_types = self.nodes.last().unwrap().output_types();

        let multiple_output = output_types.len() > 1;

        output_types.into_iter().for_each(|output| {
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
            #[allow(clippy::let_and_return)]
            pub fn forward(&self, #input_def) -> #output_type_def {
                #body

                #output_return_def
            }
        }
    }
}

#[derive(new)]
struct BurnGraphState<'a, PS: PrecisionSettings> {
    nodes: &'a Vec<Node<PS>>,
}

impl<'a, PS: PrecisionSettings> Serialize for BurnGraphState<'a, PS> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let nodes_with_names = self
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
