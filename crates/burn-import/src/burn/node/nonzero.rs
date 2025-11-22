use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorKind, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct NonZeroNode {
    pub input: TensorType,
    pub output: TensorType,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for NonZeroNode {
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;

        // Generate the appropriate zero value based on input tensor type
        let zero_value = match self.input.kind {
            TensorKind::Float => quote! { 0.0 },
            TensorKind::Int => quote! { 0 },
            TensorKind::Bool => {
                // For bool tensors, we can use argwhere directly since false is the "zero" value
                // ONNX NonZero expects output shape [rank, num_nonzero] but argwhere returns [num_nonzero, rank]
                // So we need to transpose the result
                return quote! {
                    let #output = #input.argwhere().transpose();
                };
            }
        };

        // For numeric tensors, create boolean mask and then get indices
        // ONNX NonZero expects output shape [rank, num_nonzero] but argwhere returns [num_nonzero, rank]
        // So we need to transpose the result
        quote! {
            let #output = #input.not_equal_elem(#zero_value).argwhere().transpose();
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::NonZero(self)
    }
}

impl OnnxIntoNode for NonZeroNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::NonZero(n) = node else {
            panic!("Expected NonZero node");
        };
        let input = TensorType::from(n.inputs.first().unwrap());
        let output = TensorType::from(n.outputs.first().unwrap());
        Self::new(input, output)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        TensorType,
        graph::BurnGraph,
        node::{nonzero::NonZeroNode, test::assert_tokens},
    };

    #[test]
    fn test_codegen_nonzero_float() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(NonZeroNode::new(
            TensorType::new_float("input", 2),
            TensorType::new_int("output", 2),
        ));

        graph.register_input_output(
            vec!["input".to_string()],
            vec!["output".to_string()],
            &[],
            &[],
        );

        let expected = quote! {
            use burn::prelude::*;

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
                pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2, Int> {
                    let output = input.not_equal_elem(0.0).argwhere().transpose();
                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_nonzero_int() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(NonZeroNode::new(
            TensorType::new_int("input", 2),
            TensorType::new_int("output", 2),
        ));

        graph.register_input_output(
            vec!["input".to_string()],
            vec!["output".to_string()],
            &[],
            &[],
        );

        let expected = quote! {
            use burn::prelude::*;

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
                pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 2, Int> {
                    let output = input.not_equal_elem(0).argwhere().transpose();
                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_nonzero_bool() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(NonZeroNode::new(
            TensorType::new_bool("input", 2),
            TensorType::new_int("output", 2),
        ));

        graph.register_input_output(
            vec!["input".to_string()],
            vec!["output".to_string()],
            &[],
            &[],
        );

        let expected = quote! {
            use burn::prelude::*;

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
                pub fn forward(&self, input: Tensor<B, 2, Bool>) -> Tensor<B, 2, Int> {
                    let output = input.argwhere().transpose();
                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
