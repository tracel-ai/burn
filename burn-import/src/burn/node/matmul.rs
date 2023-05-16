use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

use crate::burn::{NodeCodegen, Scope, TensorInput, TensorOutput, ToTokens};

#[derive(Debug, Clone, new)]
pub struct Matmul {
    pub lhs: TensorInput,
    pub rhs: TensorInput,
    pub output: TensorOutput,
}

impl NodeCodegen for Matmul {
    fn output_type(&self) -> TokenStream {
        let dim = self.output.dim.to_tokens();

        quote! {
            Tensor<B, #dim>
        }
    }

    fn output_name(&self) -> Ident {
        self.output.name.clone()
    }

    fn input_def(&self) -> TokenStream {
        let name_lhs = &self.lhs.name;
        let name_rhs = &self.rhs.name;
        let dim_lhs = self.lhs.dim.to_tokens();
        let dim_rhs = self.rhs.dim.to_tokens();

        quote! {
            #name_lhs: Tensor<B, #dim_lhs>, #name_rhs: Tensor<B, #dim_rhs>
        }
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let lhs = scope.use_owned_tensor(&self.lhs.name, node_position);
        let rhs = scope.use_owned_tensor(&self.rhs.name, node_position);
        let output = &self.output.name;

        quote! {
            let #output = #lhs.matmul(#rhs);
        }
    }

    fn input_tensors(&self) -> Vec<Ident> {
        vec![self.lhs.name.clone(), self.rhs.name.clone()]
    }

    fn output_tensors(&self) -> Vec<Ident> {
        vec![self.output.name.clone()]
    }

    fn field_name(&self) -> Option<Ident> {
        None
    }

    fn new_body(&self) -> TokenStream {
        quote!()
    }

    fn new_field(&self) -> TokenStream {
        quote!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::{
        graph::Graph,
        node::{conv2d::Conv2d, test::assert_tokens},
    };
    use burn::{nn::conv::Conv2dConfig, tensor::Data};
    use proc_macro2::Span;

    #[test]
    fn test_codegen() {
        let mut graph = Graph::default();

        graph.register(Matmul::new(
            TensorInput::new(Ident::new("tensor1", Span::call_site()), 4),
            TensorInput::new(Ident::new("tensor2", Span::call_site()), 4),
            TensorOutput::new(Ident::new("tensor3", Span::call_site()), 4),
        ));
        graph.register(Conv2d::new(
            Ident::new("conv2d", Span::call_site()),
            TensorInput::new(Ident::new("tensor3", Span::call_site()), 4),
            TensorOutput::new(Ident::new("tensor4", Span::call_site()), 4),
            Data::from([2.]).serialize(),
            Data::from([2.]).serialize(),
            Conv2dConfig::new([3, 3], [3, 3]),
        ));

        let expected = quote! {
            pub struct Model <B : Backend> {
                conv2d: Conv2d<B>,
            }

            impl<B: Backend> Model <B> {
                pub fn init_with (record: ModelRecord <B>) -> Self {
                    let conv2d = Conv2dConfig::new([3usize, 3usize], [3usize, 3usize])
                        .with_stride([1usize, 1usize])
                        .with_dilation([1usize, 1usize])
                        .with_groups(1usize)
                        .with_bias(true)
                        .init_with(record.conv2d);

                    Self {
                        conv2d,
                    }
                }

                pub fn forward(&self, tensor1: Tensor <B, 4>, tensor2: Tensor <B, 4>) -> Tensor <B, 4> {
                    let tensor3 = tensor1.matmul(tensor2);
                    let tensor4 = self.conv2d.forward(tensor3);

                    tensor4
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
