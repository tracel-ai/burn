use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct RandomNormalLikeNode {
    pub mean: f64,
    pub scale: f64,
    pub input: TensorType,  // Input tensor to copy shape from
    pub output: TensorType,
}

impl RandomNormalLikeNode {
    // Get shape from the input tensor
    fn get_output_shape(&self) -> TokenStream {
        let shape_it = self.input.shape.as_ref().expect("Input tensor has no shape!").iter();
        quote! { Shape::new([#(#shape_it),*]) }
    }

    // Set distribution parameters based on mean and scale
    fn get_distribution(&self) -> TokenStream {
        let mean = self.mean;
        let std_deviation = self.scale; // Scale parameter as per ONNX specs
        quote! { Distribution::Normal(#mean, #std_deviation) }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for RandomNormalLikeNode {
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]  // Input tensor type
    }

    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn forward(&self, _scope: &mut Scope, _node_position: usize) -> TokenStream {
        let output = &self.output.name;
        let shape = self.get_output_shape();
        let dist = self.get_distribution();
        quote! {
            let #output = Tensor::random(#shape, #dist, &*self.device);
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::RandomNormalLike(self)
    }

    fn register_imports(&self, imports: &mut crate::burn::BurnImports) {
        imports.register("burn::tensor::Distribution");
        imports.register("burn::prelude::Shape");
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;
    use super::*;
    use crate::burn::{
        graph::BurnGraph,
        node::test::assert_tokens,
        TensorType, TensorKind,
    };

    #[test]
    fn test_random_normal_like_codegen() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(RandomNormalLikeNode::new(
            0.0f64,
            1.0f64,
            TensorType::new("input", 2, TensorKind::Float, Some(vec![2, 3])),
            TensorType::new("output", 2, TensorKind::Float, Some(vec![2, 3])),
        ));

        graph.register_input_output(vec!["input".to_string()], vec!["output".to_string()]);

        let expected = quote! {
            use burn::prelude::Shape;
            use burn::tensor::Distribution;
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
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
                    let output = Tensor::random(
                        Shape::new([2usize, 3usize]),
                        Distribution::Normal(0f64, 1f64),
                        &*self.device,
                    );
                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
