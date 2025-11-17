use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorType, ToTokens, Type};
use burn::record::PrecisionSettings;
use onnx_ir::Argument;
use proc_macro2::TokenStream;
use quote::quote;

/// Burn-import version of ExpandConfig that stores Argument instead of RuntimeInputRef
#[derive(Debug, Clone)]
pub enum ExpandConfig {
    Static(Vec<i64>),
    Runtime(Argument),
}

#[derive(Debug, Clone, new)]
pub struct ExpandNode {
    pub input: TensorType,
    pub output: TensorType,
    pub shape: ExpandConfig,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for ExpandNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<Type> {
        let input = Type::Tensor(self.input.clone());
        // If the shape is static, we only have the input tensor as an input,
        // if it is dynamic, the shape will be our 2nd:
        match &self.shape {
            ExpandConfig::Static(_) => vec![input],
            ExpandConfig::Runtime(rt_type) => vec![input, Type::from(rt_type)],
        }
    }
    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;
        let output_rank = &self.output.rank;

        let shape = match &self.shape {
            ExpandConfig::Static(static_shape) => static_shape.to_tokens(),
            ExpandConfig::Runtime(ty) => match Type::from(ty) {
                Type::Tensor(shape_tensor) => {
                    let tensor_name = &shape_tensor.name;
                    quote! {
                        TryInto::<[B::IntElem; #output_rank]>::try_into(#tensor_name.to_data().as_slice::<B::IntElem>().unwrap()).unwrap()
                    }
                }
                Type::Shape(shape) => {
                    // Shape arrays are [i64; N] and expand now accepts them directly via Element trait
                    let shape_name = &shape.name;
                    quote! { #shape_name }
                }
                b => panic!("Invalid shape source {b:?}"),
            },
        };

        quote! {
            let #output = #input.expand(#shape);
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Expand(self)
    }
}

impl OnnxIntoNode for ExpandNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::Expand(n) = node else {
            panic!("Expected Expand node");
        };
        let input = TensorType::from(n.inputs.first().unwrap());
        let output = TensorType::from(n.outputs.first().unwrap());

        // Convert from onnx-ir ExpandConfig (with RuntimeInputRef) to burn-import ExpandConfig (with Argument)
        let shape = match n.config {
            onnx_ir::node::expand::ExpandConfig::Static(s) => ExpandConfig::Static(s),
            onnx_ir::node::expand::ExpandConfig::Runtime(shape_ref) => {
                // Get the actual argument using the RuntimeInputRef
                let shape_arg = n.inputs[shape_ref.input_index].clone();
                ExpandConfig::Runtime(shape_arg)
            }
        };
        Self::new(input, output, shape)
    }
}
