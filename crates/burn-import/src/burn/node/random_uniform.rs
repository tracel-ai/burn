use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone)]
pub struct RandomUniformNode {
    pub low: f64,
    pub high: f64,
    pub output_ty: TensorType,
    pub shape: Vec<usize>,
}

impl RandomUniformNode {
    pub fn new(output_ty: TensorType, low: f64, high: f64, shape: Vec<usize>) -> Self {
        Self {
            low,
            high,
            output_ty,
            shape,
        }
    }

    fn get_output_shape(&self) -> TokenStream {
        let shape_it = self.shape.iter();
        quote! { Shape::new([#(#shape_it),*]) }
    }

    fn get_distribution(&self) -> TokenStream {
        let low = self.low;
        let high = self.high;
        quote! { Distribution::Uniform(#low, #high) }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for RandomUniformNode {
    fn input_types(&self) -> Vec<Type> {
        Vec::with_capacity(0)
    }

    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output_ty.clone())]
    }

    fn forward(&self, _scope: &mut Scope, _node_position: usize) -> TokenStream {
        let output = &self.output_ty.name;
        let shape = self.get_output_shape();
        let dist = self.get_distribution();
        quote! {
            let #output = Tensor::random(#shape, #dist, &*self.device);
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::RandomUniform(self)
    }

    fn register_imports(&self, imports: &mut crate::burn::BurnImports) {
        imports.register("burn::tensor::Distribution");
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        TensorKind, TensorType,
        graph::BurnGraph,
        node::{random_uniform::RandomUniformNode, test::assert_tokens},
    };

    #[test]
    fn test_codegen_nodes() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(RandomUniformNode::new(
            TensorType::new("tensor1", 2, TensorKind::Float),
            0.0f64,
            1.0f64,
            vec![2, 3],
        ));

        graph.register_input_output(vec![], vec!["tensor1".to_string()]);

        let expected = quote! {
            use burn::prelude::*;
            use burn::tensor::Distribution;

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self) -> Tensor<B, 2> {
                    let tensor1 = Tensor::random(
                        Shape::new([2usize, 3usize]),
                        Distribution::Uniform(0f64, 1f64),
                        &*self.device,
                    );

                    tensor1
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
