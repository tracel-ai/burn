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
