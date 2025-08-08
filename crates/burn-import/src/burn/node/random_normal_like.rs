use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct RandomNormalLikeNode {
    pub mean: f64,
    pub scale: f64,
    pub input: TensorType,
    pub output: TensorType,
}

impl RandomNormalLikeNode {
    // Set distribution parameters based on mean and scale
    fn get_distribution(&self) -> TokenStream {
        let mean = self.mean;
        let std_deviation = self.scale;
        quote! { Distribution::Normal(#mean, #std_deviation) }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for RandomNormalLikeNode {
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let output = &self.output.name;
        let input = scope.tensor_use_owned(&self.input, node_position);
        let dist = self.get_distribution();
        quote! {
            let #output = #input.random_like(#dist);
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::RandomNormalLike(self)
    }

    fn register_imports(&self, imports: &mut crate::burn::BurnImports) {
        imports.register("burn::tensor::Distribution");
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::burn::{TensorKind, TensorType, graph::BurnGraph, node::test::assert_tokens};
    use burn::record::FullPrecisionSettings;

    #[test]
    fn test_random_normal_like_codegen() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(RandomNormalLikeNode::new(
            0.0f64,
            1.0f64,
            TensorType::new("input", 2, TensorKind::Float),
            TensorType::new("output", 2, TensorKind::Float),
        ));

        graph.register_input_output(vec!["input".to_string()], vec!["output".to_string()]);

        let expected = quote! {
            use burn::tensor::Distribution;
            use burn::tensor::Tensor;
            use burn::{
                module::Module,
                tensor::backend::Backend,
            };

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model<B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
                    let output = input.random_like(Distribution::Normal(0f64, 1f64));
                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
