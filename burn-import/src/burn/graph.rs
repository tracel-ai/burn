use super::Node;
use proc_macro2::TokenStream;
use quote::quote;
use std::collections::HashMap;

#[derive(Debug, Clone, Hash)]
pub struct NodeId(String);

#[derive(Default, Debug, Clone)]
pub struct Graph {
    var_count: HashMap<String, usize>,
    nodes: Vec<Node>,
}

impl Graph {
    /// The node must be registered in the same order they will be executed in the forward pass.
    pub fn register(&mut self, mut node: Node) {
        self.nodes.push(node);
    }

    pub fn gen_all(&self) -> TokenStream {
        let gen_struct = self.gen_struct();
        let gen_init = self.gen_init_fn();
        let gen_forward = self.gen_forward();

        quote! {
            #gen_struct

            impl<B: Backend> Model<B> {
                #gen_init

                #gen_forward
            }
        }
    }

    pub fn gen_struct(&self) -> TokenStream {
        let mut body = quote! {};
        self.nodes
            .iter()
            .map(|node| node.gen_model_field())
            .for_each(|code| body.extend(code));

        quote! {
            pub struct Model<B: Backend> {
                #body
            }
        }
    }

    pub fn gen_init_fn(&self) -> TokenStream {
        let mut body = quote! {};

        self.nodes
            .iter()
            .map(|node| node.init_with_body())
            .for_each(|code| body.extend(code));

        let fields = self
            .nodes
            .iter()
            .flat_map(|node| node.field_name())
            .collect::<Vec<_>>();

        quote! {
            pub fn init_with(record: ModelRecord<B>) -> Self {
                #body

                Self {
                    #(#fields,)*
                }
            }
        }
    }

    pub fn gen_forward(&self) -> TokenStream {
        let inputs = self.nodes.get(0).unwrap().input_definition();
        let return_type = self.nodes.get(0).unwrap().output_type();
        let output_name = self.nodes.get(0).unwrap().output_type();

        let mut body = quote! {};
        self.nodes
            .iter()
            .map(|node| node.gen_model_forward())
            .for_each(|code| body.extend(code));

        quote! {
            pub fn forward(&self, #inputs) -> #return_type {
                #body

                #output_name
            }
        }
    }
}
