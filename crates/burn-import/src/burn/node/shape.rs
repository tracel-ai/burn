use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{BurnImports, Scope, ShapeType, ToTokens, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct ShapeNode {
    pub input: Type,
    pub output: ShapeType,
    pub start_dim: usize,
    pub end_dim: usize,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for ShapeNode {
    fn input_types(&self) -> Vec<Type> {
        vec![self.input.clone()]
    }

    fn output_types(&self) -> Vec<Type> {
        vec![Type::Shape(self.output.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let output = &self.output.name;
        let dim = self.output.rank.to_tokens();
        let start_dim_tok = self.start_dim.to_tokens();
        let end_dim_tok = self.end_dim.to_tokens();

        let function = match &self.input {
            Type::Tensor(tensor) => {
                let input = scope.tensor_use_owned(tensor, node_position);
                quote! {
                    #input.dims()[#start_dim_tok..#end_dim_tok]
                        .iter()
                        .map(|&x| x as i64)
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap()
                }
            }
            Type::Shape(shape_type) => {
                // If input is already a shape array [i64; N], the Shape operation
                // returns the dimensionality of the shape (which is N) as a Shape(1) array
                // This matches the ONNX semantics where Shape of a shape gives you the rank
                let rank_value = shape_type.rank as i64;
                quote! { [#rank_value] }
            }
            _ => panic!("Shape operation only supports Tensor or Shape inputs"),
        };

        quote! {
            let #output: [i64;#dim] = #function;
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Shape(self)
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("alloc::vec::Vec");
    }
}

impl OnnxIntoNode for ShapeNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::Shape(n) = node else {
            panic!("Expected Shape node");
        };
        let input = Type::from(n.inputs.first().unwrap());
        let output = match Type::from(n.outputs.first().unwrap()) {
            Type::Shape(s) => s,
            _ => panic!("Shape expects shape output"),
        };
        let start_dim = n.config.start;
        let end_dim = n.config.end;
        Self::new(input, output, start_dim, end_dim)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{TensorType, graph::BurnGraph, node::test::assert_tokens};

    #[test]
    fn test_codegen_shape() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ShapeNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 4)),
            ShapeType::new("shape1", 2),
            1,
            3,
        ));

        graph.register_input_output(
            vec!["tensor1".to_string()],
            vec!["shape1".to_string()],
            &[],
            &[],
        );

        let expected = quote! {
            use burn::prelude::*;
            use alloc::vec::Vec;

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
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> [i64; 2] {
                    let shape1: [i64; 2] = tensor1.dims()[1..3]
                        .iter()
                        .map(|&x| x as i64)
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap();
                    shape1
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
