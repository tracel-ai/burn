use super::{BurnImports, Scope, Type};
use crate::burn::{
    node::{Node, NodeCodegen},
    ToTokens,
};
use burn::record::{BurnRecord, FileRecorder, PrecisionSettings};
use proc_macro2::TokenStream;
use quote::quote;
use serde::{ser::SerializeMap, Serialize};
use std::path::PathBuf;

#[derive(Default, Debug)]
pub struct Graph<PS: PrecisionSettings> {
    scope: Scope,
    imports: BurnImports,
    nodes: Vec<Node<PS>>,
}

impl<PS: PrecisionSettings> Serialize for Graph<PS> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let nodes_with_names = self
            .nodes
            .iter()
            .filter_map(|node| {
                if let Some(name) = node.field_name() {
                    Some((node, name))
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

impl<PS: PrecisionSettings> Graph<PS> {
    /// The node must be registered in the same order they will be executed in the forward pass.
    pub fn register<N: NodeCodegen<PS> + 'static>(&mut self, node: N) {
        self.nodes.push(node.into_node());
    }

    fn build_scope(&mut self) {
        let input = self.nodes.first().unwrap();
        let to_tensor_ident = |ty: Type| match ty {
            super::Type::Tensor(ty) => Some(ty.name),
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

    pub fn save_record<FR: FileRecorder>(&self, recorder: FR, file: PathBuf) {
        recorder
            .save_item(BurnRecord::new::<FR>(self), file)
            .unwrap();
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

        quote! {
            #codegen_imports
            #codegen_struct

            impl<B: Backend> Model<B> {
                #codegen_init

                #codegen_forward
            }
        }
    }

    fn codegen_struct(&self) -> TokenStream {
        let mut body = quote! {};
        self.nodes
            .iter()
            .map(|node| node.new_field())
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
            .map(|node| node.new_body())
            .for_each(|code| body.extend(code));

        let fields = self
            .nodes
            .iter()
            .flat_map(|node| node.field_name())
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
            .for_each(|ty| match ty {
                Type::Tensor(tensor) => {
                    let name = &tensor.name;
                    let dim = tensor.dim.to_tokens();
                    input_def.extend(quote! {
                            #name: Tensor<B, #dim>,

                    })
                }
                Type::Other(other) => {
                    let name = &other.name;
                    let ty = &other.ty;

                    input_def.extend(quote! {
                        #name: #ty,

                    })
                }
            });

        let output_types = self.nodes.last().unwrap().output_types();

        let multiple_output = output_types.len() > 1;

        output_types.into_iter().for_each(|ty| match ty {
            Type::Tensor(tensor) => {
                let name = &tensor.name;
                let dim = tensor.dim.to_tokens();

                if multiple_output {
                    output_type_def.extend(quote! {
                        Tensor<B, #dim>,
                    });
                    output_return_def.extend(quote! {
                        #name,
                    });
                } else {
                    output_type_def.extend(quote! {
                        Tensor<B, #dim>
                    });
                    output_return_def.extend(quote! {
                        #name
                    });
                }
            }
            Type::Other(other) => {
                let name = &other.name;
                let ty = other.ty;

                if multiple_output {
                    output_return_def.extend(quote! {
                        #ty,
                    });
                    output_return_def.extend(quote! {
                        #name,
                    });
                } else {
                    output_return_def.extend(quote! {
                        #ty
                    });
                    output_return_def.extend(quote! {
                        #name
                    });
                }
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
