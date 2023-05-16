use super::Scope;
use crate::burn::NodeCodegen;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, Hash)]
pub struct NodeId(String);

#[derive(Default, Debug)]
pub struct Graph {
    scope: Scope,
    nodes: Vec<Box<dyn NodeCodegen>>,
}

impl Graph {
    /// The node must be registered in the same order they will be executed in the forward pass.
    pub fn register<N: NodeCodegen + 'static>(&mut self, node: N) {
        self.nodes.push(Box::new(node));
    }

    fn build_scope(&mut self) {
        let input = self.nodes.first().unwrap();
        input
            .input_tensors()
            .iter()
            .for_each(|tensor| self.scope.declare_tensor(tensor, 0));
        println!("{:?}", self.scope);

        self.nodes
            .iter()
            .enumerate()
            .for_each(|(node_position, node)| {
                node.output_tensors()
                    .iter()
                    .for_each(|tensor| self.scope.declare_tensor(tensor, node_position + 1))
            });

        self.nodes
            .iter()
            .enumerate()
            .for_each(|(node_position, node)| {
                node.input_tensors()
                    .iter()
                    .for_each(|tensor| self.scope.register_use_owned_tensor(tensor, node_position))
            });
    }

    pub fn codegen(mut self) -> TokenStream {
        self.build_scope();

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

    fn gen_struct(&self) -> TokenStream {
        let mut body = quote! {};
        self.nodes
            .iter()
            .map(|node| node.new_field())
            .for_each(|code| body.extend(code));

        quote! {
            pub struct Model<B: Backend> {
                #body
            }
        }
    }

    fn gen_init_fn(&self) -> TokenStream {
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
            pub fn init_with(record: ModelRecord<B>) -> Self {
                #body

                Self {
                    #(#fields,)*
                }
            }
        }
    }

    fn gen_forward(&mut self) -> TokenStream {
        let inputs = self.nodes.first().unwrap().input_def();
        let output_type = self.nodes.last().unwrap().output_type();
        let output_name = self.nodes.last().unwrap().output_name();

        let mut body = quote! {};
        self.nodes
            .iter()
            .enumerate()
            .map(|(index, node)| node.forward(&mut self.scope, index))
            .for_each(|code| body.extend(code));

        quote! {
            pub fn forward(&self, #inputs) -> #output_type {
                #body

                #output_name
            }
        }
    }
}
