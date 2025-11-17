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
        let onnx_ir::Node::Concat(n) = node else {
            panic!("Expected Concat node");
        };
        let inputs: Vec<Type> = n.inputs.iter().map(Type::from).collect();
        let output = Type::from(n.outputs.first().unwrap());
        let dim = n.config.axis;
        Self::new(inputs, output, dim)
    }
}
