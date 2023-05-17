use super::{conv2d::Conv2dNode, linear::LinearNode, matmul::MatmulNode};
use crate::burn::{BurnImports, Scope};
use burn::record::PrecisionSettings;
use burn_ndarray::NdArrayBackend;
use proc_macro2::{Ident, TokenStream};
use quote::quote;
use serde::Serialize;

pub type SerializationBackend = NdArrayBackend<f32>;

pub trait NodeCodegen<PS: PrecisionSettings>: std::fmt::Debug + Serialize {
    fn output_type(&self) -> TokenStream;
    fn output_name(&self) -> Ident;
    fn input_def(&self) -> TokenStream;
    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream;

    fn field_name(&self) -> Option<Ident> {
        None
    }
    fn new_body(&self) -> TokenStream {
        quote! {}
    }
    fn new_field(&self) -> TokenStream {
        quote! {}
    }
    fn input_tensors(&self) -> Vec<Ident> {
        vec![]
    }
    fn output_tensors(&self) -> Vec<Ident> {
        vec![]
    }
    fn register_imports(&self, _imports: &mut BurnImports) {}
    fn into_node(self) -> Node<PS>;
}

#[derive(Debug)]
pub enum Node<PS: PrecisionSettings> {
    Matmul(MatmulNode),
    Conv2d(Conv2dNode<PS>),
    Linear(LinearNode<PS>),
}

macro_rules! match_all {
    ($self:expr, $func:expr) => {{
        match $self {
            Node::Matmul(node) => $func(node),
            Node::Conv2d(node) => $func(node),
            Node::Linear(node) => $func(node),
        }
    }};
}

impl<PS: PrecisionSettings> Serialize for Node<PS> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match_all!(self, |node| Serialize::serialize::<S>(node, serializer))
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for Node<PS> {
    fn output_type(&self) -> TokenStream {
        match_all!(self, NodeCodegen::<PS>::output_type)
    }

    fn output_name(&self) -> Ident {
        match_all!(self, NodeCodegen::<PS>::output_name)
    }

    fn input_def(&self) -> TokenStream {
        match_all!(self, NodeCodegen::<PS>::input_def)
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        match_all!(self, |node| NodeCodegen::<PS>::forward(
            node,
            scope,
            node_position
        ))
    }

    fn field_name(&self) -> Option<Ident> {
        match_all!(self, NodeCodegen::<PS>::field_name)
    }

    fn new_body(&self) -> TokenStream {
        match_all!(self, NodeCodegen::<PS>::new_body)
    }

    fn new_field(&self) -> TokenStream {
        match_all!(self, NodeCodegen::<PS>::new_field)
    }

    fn input_tensors(&self) -> Vec<Ident> {
        match_all!(self, NodeCodegen::<PS>::input_tensors)
    }

    fn output_tensors(&self) -> Vec<Ident> {
        match_all!(self, NodeCodegen::<PS>::output_tensors)
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        match_all!(self, |node| NodeCodegen::<PS>::register_imports(
            node, imports
        ))
    }

    fn into_node(self) -> Node<PS> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::{
        graph::Graph,
        node::{conv2d::Conv2dNode, matmul::MatmulNode, test::assert_tokens},
        TensorDescription,
    };
    use burn::{nn::conv::Conv2dConfig, record::FullPrecisionSettings, tensor::Data};
    use proc_macro2::Span;

    #[test]
    fn test_codegen_two_nodes() {
        let mut graph = Graph::<FullPrecisionSettings>::default();

        graph.register(MatmulNode::new(
            TensorDescription::new("tensor1", 4),
            TensorDescription::new("tensor2", 4),
            TensorDescription::new("tensor3", 4),
        ));
        graph.register(Conv2dNode::new(
            Ident::new("conv2d", Span::call_site()),
            TensorDescription::new("tensor3", 4),
            TensorDescription::new("tensor4", 4),
            Data::from([2.]).serialize(),
            None,
            Conv2dConfig::new([3, 3], [3, 3]),
        ));

        let expected = quote! {
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };
            use burn::nn::conv::Conv2d;
            use burn::nn::conv::Conv2dConfig;

            #[derive(Module, Debug)]
            pub struct Model <B: Backend> {
                conv2d: Conv2d<B>,
            }

            impl<B: Backend> Model <B> {
                pub fn new_with(record: ModelRecord<B>) -> Self {
                    let conv2d = Conv2dConfig::new([3, 3], [3, 3])
                        .with_stride([1, 1])
                        .with_dilation([1, 1])
                        .with_groups(1)
                        .with_bias(true)
                        .init_with(record.conv2d);

                    Self {
                        conv2d,
                    }
                }

                pub fn forward(&self, tensor1: Tensor<B, 4>, tensor2: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor3 = tensor1.matmul(tensor2);
                    let tensor4 = self.conv2d.forward(tensor3);

                    tensor4
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_clone_tensor() {
        let mut graph = Graph::<FullPrecisionSettings>::default();

        graph.register(MatmulNode::new(
            TensorDescription::new("tensor1", 4),
            TensorDescription::new("tensor2", 4),
            TensorDescription::new("tensor3", 4),
        ));
        graph.register(Conv2dNode::new(
            Ident::new("conv2d", Span::call_site()),
            TensorDescription::new("tensor2", 4),
            TensorDescription::new("tensor4", 4),
            Data::from([2.]).serialize(),
            None,
            Conv2dConfig::new([3, 3], [3, 3]),
        ));
        graph.register(MatmulNode::new(
            TensorDescription::new("tensor3", 4),
            TensorDescription::new("tensor4", 4),
            TensorDescription::new("output", 4),
        ));

        let expected = quote! {
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };
            use burn::nn::conv::Conv2d;
            use burn::nn::conv::Conv2dConfig;

            #[derive(Module, Debug)]
            pub struct Model <B: Backend> {
                conv2d: Conv2d<B>,
            }

            impl<B: Backend> Model <B> {
                pub fn new_with(record: ModelRecord<B>) -> Self {
                    let conv2d = Conv2dConfig::new([3, 3], [3, 3])
                        .with_stride([1, 1])
                        .with_dilation([1, 1])
                        .with_groups(1)
                        .with_bias(true)
                        .init_with(record.conv2d);

                    Self {
                        conv2d,
                    }
                }

                pub fn forward(&self, tensor1: Tensor<B, 4>, tensor2: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor3 = tensor1.matmul(tensor2.clone());
                    let tensor4 = self.conv2d.forward(tensor2);
                    let output = tensor3.matmul(tensor4);

                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
