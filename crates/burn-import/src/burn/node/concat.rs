use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, ToTokens, Type};

use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct ConcatNode {
    pub inputs: Vec<Type>,
    pub output: Type,
    pub dim: usize,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for ConcatNode {
    fn output_types(&self) -> Vec<Type> {
        vec![self.output.clone()]
    }

    fn input_types(&self) -> Vec<Type> {
        self.inputs.clone()
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        match &self.output {
            Type::Tensor(output_tensor) => {
                // Tensor concatenation
                let dim = self.dim.to_tokens();
                let inputs = self.inputs.iter().map(|t| match t {
                    Type::Tensor(tensor) => scope.tensor_use_owned(tensor, node_position),
                    _ => panic!("Expected tensor input for tensor concatenation"),
                });

                let output = &output_tensor.name;

                quote! {
                    let #output = burn::tensor::Tensor::cat([#(#inputs),*].into(), #dim);
                }
            }
            Type::Shape(shape_type) => {
                // Shape concatenation - shapes are 1D so concat is always on axis 0
                if self.dim != 0 {
                    panic!(
                        "Shape concatenation only supports dim=0, got dim={}",
                        self.dim
                    );
                }
                let output = &shape_type.name;
                let output_rank = shape_type.rank;

                // Generate code to concatenate shape arrays
                let mut shape_parts = Vec::new();
                for input in &self.inputs {
                    match input {
                        Type::Shape(shape) => {
                            let input_name = &shape.name;
                            shape_parts.push(quote! { &#input_name[..] });
                        }
                        _ => panic!("Expected shape input for shape concatenation"),
                    }
                }

                quote! {
                    let #output: [i64; #output_rank] = [#(#shape_parts),*].concat().try_into().unwrap();
                }
            }
            _ => panic!("Concat only supports Tensor or Shape outputs"),
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Concat(self)
    }
}

impl OnnxIntoNode for ConcatNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let (inputs_vec, outputs, config) = match node {
            onnx_ir::Node::Concat {
                inputs,
                outputs,
                config,
                ..
            } => (inputs, outputs, config),
            _ => panic!("Expected Concat node"),
        };
        let inputs: Vec<Type> = inputs_vec.iter().map(Type::from).collect();
        let output = Type::from(outputs.first().unwrap());
        let dim = config.axis;
        Self::new(inputs, output, dim)
    }
}

#[cfg(test)]
mod tests {

    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        TensorType, Type,
        graph::BurnGraph,
        node::{concat::ConcatNode, test::assert_tokens},
    };

    #[test]
    fn test_codegen_concat_tensors() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ConcatNode::new(
            vec![
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_float("tensor2", 4)),
            ],
            Type::Tensor(TensorType::new_float("tensor3", 4)),
            1,
        ));

        graph.register_input_output(
            vec!["tensor1".to_string(), "tensor2".to_string()],
            vec!["tensor3".to_string()],
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
                    tensor1: Tensor<B, 4>,
                    tensor2: Tensor<B, 4>
                ) -> Tensor<B, 4> {
                    let tensor3 = burn::tensor::Tensor::cat([tensor1, tensor2].into(), 1);

                    tensor3
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_concat_shapes() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ConcatNode::new(
            vec![
                Type::Shape(crate::burn::ShapeType::new("shape1", 2)),
                Type::Shape(crate::burn::ShapeType::new("shape2", 3)),
                Type::Shape(crate::burn::ShapeType::new("shape3", 1)),
            ],
            Type::Shape(crate::burn::ShapeType::new("output_shape", 6)),
            0,
        ));

        graph.register_input_output(
            vec![
                "shape1".to_string(),
                "shape2".to_string(),
                "shape3".to_string(),
            ],
            vec!["output_shape".to_string()],
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
                    shape1: [i64; 2],
                    shape2: [i64; 3],
                    shape3: [i64; 1]
                ) -> [i64; 6] {
                    let output_shape: [i64; 6usize] = [&shape1[..], &shape2[..], &shape3[..]].concat().try_into().unwrap();

                    output_shape
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
