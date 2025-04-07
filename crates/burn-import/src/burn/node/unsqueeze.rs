use super::{Node, NodeCodegen};
use crate::burn::{BurnImports, Scope, TensorType, ToTokens, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct UnsqueezeNode {
    pub input: Type,
    pub output: TensorType,
    pub axes: Vec<i64>,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for UnsqueezeNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![self.input.clone()]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let output = &self.output.name;
        let shape_values = &self.axes.to_tokens();
        let new_dims = self.output.rank.to_tokens();

        match &self.input {
            Type::Tensor(tensor) => {
                let input = scope.tensor_use_owned(tensor, node_position);
                quote! {
                    let #output: Tensor<B, #new_dims> = #input.unsqueeze_dims(&#shape_values);
                }
            }
            Type::Scalar(scalar) => {
                let input = &scalar.name;
                quote! {
                    let #output = Tensor::<B, #new_dims>::from_data([#input.elem::<B::FloatElem>()], &self.device).unsqueeze();
                }
            }
            _ => panic!("Unsupported input type"),
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Unsqueeze(self)
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        match &self.input {
            Type::Scalar(_) => {
                imports.register("burn::tensor::ElementConversion");
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        TensorType, Type,
        graph::BurnGraph,
        node::{test::assert_tokens, unsqueeze::UnsqueezeNode},
    };

    #[test]
    fn test_codegen_nodes() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(UnsqueezeNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 3)),
            TensorType::new_float("tensor2", 5),
            [0, 4].into(),
        ));

        graph.register_input_output(vec!["tensor1".to_string()], vec!["tensor2".to_string()]);

        let expected = quote! {
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };

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
                pub fn forward(&self, tensor1: Tensor<B, 3>) -> Tensor<B, 5> {
                    let tensor2: Tensor<B, 5> = tensor1.unsqueeze_dims(&[0,4]);
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
