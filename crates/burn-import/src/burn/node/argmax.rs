use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{TensorKind, TensorType, ToTokens, Type};

use burn::record::PrecisionSettings;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct ArgMaxNode {
    pub input: TensorType,
    pub output: Type,
    pub axis: usize,
    pub keepdims: bool,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for ArgMaxNode {
    fn output_types(&self) -> Vec<Type> {
        match &self.output {
            Type::Tensor(tensor) => {
                let mut tensor = tensor.clone();
                tensor.kind = TensorKind::Int;
                vec![Type::Tensor(tensor)]
            }
            Type::Scalar(scalar) => {
                // For scalar output, we keep the name but change to backend Int type
                // The actual type will be B::IntElem in the generated code
                vec![Type::Scalar(scalar.clone())]
            }
            _ => panic!("ArgMax output must be Tensor or Scalar"),
        }
    }

    fn input_types(&self) -> Vec<crate::burn::Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn forward(
        &self,
        scope: &mut crate::burn::Scope,
        node_position: usize,
    ) -> proc_macro2::TokenStream {
        //NOTE: select_last_index=1 is not supported (will panic during conversion)
        let axis = self.axis.to_tokens();
        let input = scope.tensor_use_owned(&self.input, node_position);

        match &self.output {
            Type::Tensor(tensor) => {
                let output = &tensor.name;
                if self.keepdims {
                    // keepdims=True: Burn's argmax keeps dimensions by default
                    quote! {
                        let #output = #input.argmax(#axis);
                    }
                } else {
                    // keepdims=False: use argmax followed by squeeze to remove the kept dimension
                    let output_rank = tensor.rank;
                    quote! {
                        let argmax_result = #input.argmax(#axis);
                        let #output = argmax_result.squeeze_dim::<#output_rank>(#axis);
                    }
                }
            }
            Type::Scalar(scalar) => {
                let output = &scalar.name;
                // 1D tensor with keepdims=false -> scalar output
                // ArgMax always outputs Int64 indices
                quote! {
                    let argmax_result = #input.argmax(#axis);
                    let #output = argmax_result.into_scalar().elem::<i64>();
                }
            }
            _ => panic!("ArgMax output must be Tensor or Scalar"),
        }
    }

    fn into_node(self) -> super::Node<PS> {
        Node::ArgMax(self)
    }
}

impl OnnxIntoNode for ArgMaxNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let input = crate::burn::TensorType::from(node.inputs.first().unwrap());
        let output = crate::burn::Type::from(node.outputs.first().unwrap());
        let config = onnx_ir::node::argmax::argmax_config(&node);
        Self::new(input, output, config.axis, config.keepdims)
    }
}

#[cfg(test)]
mod tests {

    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{TensorType, graph::BurnGraph, node::test::assert_tokens};

    #[test]
    fn test_codegen_argmax() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ArgMaxNode::new(
            TensorType::new_float("tensor1", 2),
            Type::Tensor(TensorType::new_int("tensor2", 2)),
            1,
            true, // keepdims=true
        ));

        graph.register_input_output(vec!["tensor1".to_string()], vec!["tensor2".to_string()]);

        let expected = quote! {
            use burn::prelude::*;

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
                pub fn forward(
                    &self,
                    tensor1: Tensor<B, 2>
                ) -> Tensor<B, 2, Int> {
                    let tensor2 = tensor1.argmax(1);

                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_argmax_keepdims_false() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ArgMaxNode::new(
            TensorType::new_float("tensor1", 2),
            Type::Tensor(TensorType::new_int("tensor2", 1)), // Output rank reduced due to keepdims=false
            1,
            false, // keepdims=false
        ));

        graph.register_input_output(vec!["tensor1".to_string()], vec!["tensor2".to_string()]);

        let expected = quote! {
            use burn::prelude::*;

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
                pub fn forward(
                    &self,
                    tensor1: Tensor<B, 2>
                ) -> Tensor<B, 1, Int> {
                    let argmax_result = tensor1.argmax(1);
                    let tensor2 = argmax_result.squeeze_dim::<1usize>(1);

                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
