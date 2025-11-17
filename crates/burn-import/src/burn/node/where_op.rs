use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{ScalarType, ShapeType, Type};

use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct WhereNode {
    /// Bool tensor. When True (nonzero), yield X, otherwise yield Y.
    pub condition: Type,
    /// Values selected at indices where condition is True.
    pub x: Type,
    /// Values selected at indices where condition is False.
    pub y: Type,
    pub output: Type,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for WhereNode {
    fn output_types(&self) -> Vec<Type> {
        vec![self.output.clone()]
    }

    fn input_types(&self) -> Vec<crate::burn::Type> {
        vec![self.condition.clone(), self.x.clone(), self.y.clone()]
    }

    fn forward(&self, scope: &mut crate::burn::Scope, node_position: usize) -> TokenStream {
        match &self.output {
            Type::Tensor(out) => {
                let cond = Self::input_as_tensor(&self.condition, out.rank, scope, node_position);
                let y = Self::input_as_tensor(&self.y, out.rank, scope, node_position);
                let out_id = &out.name;

                if let Type::Scalar(x) = &self.x {
                    let x = &x.name;
                    quote! {
                        let #out_id = #y.mask_fill(#cond, #x);
                    }
                } else {
                    let x = Self::input_as_tensor(&self.x, out.rank, scope, node_position);
                    quote! {
                        let #out_id = #y.mask_where(#cond, #x);
                    }
                }
            }
            Type::Scalar(out) => {
                // Scalar out means all inputs are scalars as well:
                let cond = self.condition.as_scalar();
                let x = self.x.as_scalar();
                let y = self.y.as_scalar();
                Self::forward_scalar(out, cond, x, y)
            }
            Type::Shape(out) => {
                // Shape output - all inputs should be shapes or compatible types
                Self::forward_shape(out, &self.condition, &self.x, &self.y)
            }
            other => panic!("Where cannot handle {other:?}"),
        }
    }

    fn into_node(self) -> super::Node<PS> {
        Node::Where(self)
    }
}

impl OnnxIntoNode for WhereNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::Where(n) = node else {
            panic!("Expected Where node");
        };
        let condition = Type::from(n.inputs.first().unwrap());
        let x = Type::from(n.inputs.get(1).unwrap());
        let y = Type::from(n.inputs.get(2).unwrap());
        let output = Type::from(n.outputs.first().unwrap());
        Self::new(condition, x, y, output)
    }
}

impl WhereNode {
    fn forward_scalar(
        out: &ScalarType,
        cond: &ScalarType,
        x: &ScalarType,
        y: &ScalarType,
    ) -> TokenStream {
        let out_name = &out.name;
        let out_type = out.ty();
        let cond_name = &cond.name;
        let x_name = &x.name;
        let y_name = &y.name;

        quote! {
            let #out_name : #out_type = if #cond_name {
                #x_name
            }
            else {
                #y_name
            };
        }
    }

    fn forward_shape(out: &ShapeType, condition: &Type, x: &Type, y: &Type) -> TokenStream {
        let out_name = &out.name;

        // Generate code based on input types - only semantically valid combinations
        match (condition, x, y) {
            (Type::Shape(cond), Type::Shape(x_shape), Type::Shape(y_shape)) => {
                // All shapes: element-wise selection between shape dimensions
                // Each element of condition determines whether to take from x or y
                let cond_name = &cond.name;
                let x_name = &x_shape.name;
                let y_name = &y_shape.name;

                quote! {
                    let #out_name = {
                        let mut result = #y_name;
                        for (i, (cond_item, x_item)) in #cond_name.iter().zip(#x_name.iter()).enumerate() {
                            if *cond_item != 0 {
                                result[i] = *x_item;
                            }
                        }
                        result
                    };
                }
            }
            (Type::Scalar(cond), Type::Shape(x_shape), Type::Shape(y_shape)) => {
                // Scalar condition: select entire shape x or y
                let cond_name = &cond.name;
                let x_name = &x_shape.name;
                let y_name = &y_shape.name;

                quote! {
                    let #out_name = if #cond_name { #x_name } else { #y_name };
                }
            }
            _ => panic!(
                "Where with Shape output only supports: \
                 (Shape, Shape, Shape) for element-wise selection or \
                 (Scalar, Shape, Shape) for whole shape selection"
            ),
        }
    }

    fn input_as_tensor(
        input: &Type,
        broadcast_rank: usize,
        scope: &mut crate::burn::Scope,
        node_position: usize,
    ) -> TokenStream {
        let (tensor, rank) = match input {
            Type::Tensor(t) => (scope.tensor_use_owned(t, node_position), t.rank),
            Type::Scalar(s) => (s.to_full_tensor(&vec![1; broadcast_rank]), broadcast_rank),
            Type::Shape(s) => (s.to_tensor(), 1),
            _ => panic!("Where op: {input:?} input not implemented"),
        };
        if rank < broadcast_rank {
            // Generate unsqueeze_dims to add trailing dimensions for broadcasting
            // Create a vector of dimension indices to unsqueeze at the end
            let dims_to_unsqueeze: Vec<isize> =
                (rank..broadcast_rank).map(|d| d as isize).collect();
            let dims = quote! { &[#(#dims_to_unsqueeze),*] };
            quote! { #tensor.unsqueeze_dims(#dims) }
        } else {
            tensor
        }
    }
}
