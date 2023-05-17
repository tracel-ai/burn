use crate::burn::{BurnImports, Scope};
use proc_macro2::{Ident, TokenStream};
use quote::quote;

pub trait Node: std::fmt::Debug {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::{
        graph::Graph,
        node::{conv2d::Conv2dNode, matmul::MatmulNode, test::assert_tokens},
        TensorDescription,
    };
    use burn::{nn::conv::Conv2dConfig, tensor::Data};
    use proc_macro2::Span;

    #[test]
    fn test_codegen_two_nodes() {
        let mut graph = Graph::default();

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
                pub fn init_with(record: ModelRecord<B>) -> Self {
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
        let mut graph = Graph::default();

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
                pub fn init_with(record: ModelRecord<B>) -> Self {
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
