use super::{BurnImports, Scope, Type};
use crate::burn::node::{Node, NodeCodegen};
use burn::record::{
    BurnRecord, DefaultFileRecorder, FileRecorder, PrecisionSettings, PrettyJsonFileRecorder,
};
use proc_macro2::TokenStream;
use quote::quote;
use serde::{ser::SerializeMap, Serialize};
use std::path::PathBuf;

#[derive(Default, Debug)]
pub struct BurnGraph<PS: PrecisionSettings> {
    scope: Scope,
    imports: BurnImports,
    nodes: Vec<Node<PS>>,
    top_comment: Option<String>,
    blank_spaces: bool,
    default: Option<TokenStream>,
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
            .filter_map(|node| {
                if let Some(ty) = node.field_type() {
                    Some((node, ty.name().clone()))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let mut map = serializer.serialize_map(Some(nodes_with_names.len()))?;

        for (node, name) in nodes_with_names.iter() {
            map.serialize_entry(&name.to_string(), &node)?;
        }

        map.end()
    }
}

impl<PS: PrecisionSettings> BurnGraph<PS> {
    /// The node must be registered in the same order they will be executed in the forward pass.
    pub fn register<N: NodeCodegen<PS> + 'static>(&mut self, node: N) {
        self.nodes.push(node.into_node());
        println!("Registered a node");
    }

    fn build_scope(&mut self) {
        let input = self.nodes.first().unwrap();
        let to_tensor_ident = |ty: Type| match ty {
            super::Type::Tensor(ty) => Some(ty.name.clone()),
            _ => None,
        };
        input
            .input_types()
            .into_iter()
            .flat_map(to_tensor_ident)
            .for_each(|tensor| self.scope.declare_tensor(&tensor, 0));

        self.nodes
            .iter()
            .enumerate()
            .for_each(|(node_position, node)| {
                node.output_types()
                    .into_iter()
                    .flat_map(to_tensor_ident)
                    .for_each(|tensor| self.scope.declare_tensor(&tensor, node_position + 1))
            });

        self.nodes
            .iter()
            .enumerate()
            .for_each(|(node_position, node)| {
                node.input_types()
                    .into_iter()
                    .flat_map(to_tensor_ident)
                    .for_each(|tensor| self.scope.register_use_owned_tensor(&tensor, node_position))
            });
    }

    pub fn with_record(
        mut self,
        out_file: PathBuf,
        development: bool,
        precision_ty_str: &str,
    ) -> Self {
        if development {
            let recorder = PrettyJsonFileRecorder::<PS>::new();
            self.save_record(
                recorder,
                out_file.clone(),
                &format!("burn::record::PrettyJsonFileRecorder::<{precision_ty_str}>"),
            );
        } else {
            let recorder = DefaultFileRecorder::<PS>::new();
            self.save_record(
                recorder,
                out_file.clone(),
                &format!("burn::record::DefaultFileRecorder::<{precision_ty_str}>"),
            );
        }
        self
    }

    pub fn with_blank_space(mut self, blank_spaces: bool) -> Self {
        self.blank_spaces = blank_spaces;
        self
    }

    pub fn with_top_comment(mut self, top_comment: Option<String>) -> Self {
        self.top_comment = top_comment;
        self
    }

    fn save_record<FR: FileRecorder>(&mut self, recorder: FR, file: PathBuf, recorder_str: &str) {
        self.imports.register("burn::record::Recorder");

        let state = BurnGraphState::new(&self.nodes);
        recorder
            .save_item(BurnRecord::new::<FR>(state), file.clone())
            .unwrap();

        let recorder_ty = syn::parse_str::<syn::Type>(recorder_str).unwrap();
        let file = file.to_str();

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

    pub fn codegen(mut self) -> TokenStream {
        self.build_scope();
        self.nodes
            .iter()
            .for_each(|node| node.register_imports(&mut self.imports));

        let codegen_imports = self.imports.codegen();
        let codegen_struct = self.codegen_struct();
        let codegen_init = self.codegen_new_fn();
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
            #maybe_top_file_comment
            #codegen_imports
            #maybe_blank
            #maybe_blank

            #codegen_struct
            #maybe_blank

            #codegen_default

            impl<B: Backend> Model<B> {
                #codegen_init
                #maybe_blank
                #codegen_forward
            }
        }
    }

    fn codegen_struct(&self) -> TokenStream {
        let mut body = quote! {};
        self.nodes
            .iter()
            .map(|node| node.field_type())
            .flatten()
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

    fn codegen_new_fn(&self) -> TokenStream {
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

        quote! {
            pub fn forward(&self, #input_def) -> #output_type_def {
                #body

                #output_return_def
            }
        }
    }
}
