use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{ScalarKind, ScalarType, TensorKind, TensorType, ToTokens, Type};

use burn::record::PrecisionSettings;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct ArgMinNode {
    pub input: TensorType,
    pub output: Type,
    pub axis: usize,
    pub keepdims: bool,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for ArgMinNode {
    fn output_types(&self) -> Vec<Type> {
        match &self.output {
            Type::Tensor(tensor) => {
                let mut tensor = tensor.clone();
                tensor.kind = TensorKind::Int;
                vec![Type::Tensor(tensor)]
            }
            Type::Scalar(scalar) => {
                vec![Type::Scalar(ScalarType::new(
                    scalar.name.to_string(),
                    ScalarKind::Int64,
                ))]
            }
            _ => panic!("ArgMin output must be Tensor or Scalar"),
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
                    // keepdims=True: Burn's argmin keeps dimensions by default
                    quote! {
                        let #output = #input.argmin(#axis);
                    }
                } else {
                    // keepdims=False: use argmin followed by squeeze to remove the kept dimension
                    let output_rank = tensor.rank;
                    quote! {
                        let argmin_result = #input.argmin(#axis);
                        let #output = argmin_result.squeeze_dim::<#output_rank>(#axis);
                    }
                }
            }
            Type::Scalar(scalar) => {
                let output = &scalar.name;
                // 1D tensor with keepdims=false -> scalar output
                // ArgMin always outputs Int64 indices
                quote! {
                    let argmin_result = #input.argmin(#axis);
                    let #output = argmin_result.into_scalar().elem::<i64>();
                }
            }
            _ => panic!("ArgMin output must be Tensor or Scalar"),
        }
    }

    fn into_node(self) -> super::Node<PS> {
        Node::ArgMin(self)
    }
}

impl OnnxIntoNode for ArgMinNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::ArgMin(n) = node else {
            panic!("Expected ArgMin node");
        };
        let input = crate::burn::TensorType::from(n.inputs.first().unwrap());
        let output = crate::burn::Type::from(n.outputs.first().unwrap());
        Self::new(input, output, n.config.axis, n.config.keepdims)
    }
}
