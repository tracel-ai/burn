use super::NodeCodegen;
use crate::burn::Scope;
use burn::record::PrecisionSettings;
use onnx_ir::Argument;
use proc_macro2::TokenStream;
use quote::quote;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::where_op::WhereNode {
    fn inputs(&self) -> Vec<&Argument> {
        self.inputs
            .iter()
            .filter(|arg| arg.is_dynamic() || arg.is_constant())
            .collect()
    }

    fn outputs(&self) -> Vec<&Argument> {
        self.outputs.iter().collect()
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        // Convert Arguments to Types for pattern matching
        let condition = Type::from(&self.inputs[0]);
        let x = Type::from(&self.inputs[1]);
        let y = Type::from(&self.inputs[2]);
        let output = Type::from(self.outputs.first().unwrap());

        match &output {
            Type::Tensor(out) => {
                let cond = where_input_as_tensor(&condition, out.rank, scope, node_position);
                let y_tensor = where_input_as_tensor(&y, out.rank, scope, node_position);
                let out_id = &out.name;

                if let Type::Scalar(x_scalar) = &x {
                    let x_name = &x_scalar.name;
                    quote! {
                        let #out_id = #y_tensor.mask_fill(#cond, #x_name);
                    }
                } else {
                    let x_tensor = where_input_as_tensor(&x, out.rank, scope, node_position);
                    quote! {
                        let #out_id = #y_tensor.mask_where(#cond, #x_tensor);
                    }
                }
            }
            Type::Scalar(out) => {
                // Scalar out means all inputs are scalars as well:
                let cond = condition.as_scalar();
                let x_scalar = x.as_scalar();
                let y_scalar = y.as_scalar();
                where_forward_scalar(out, cond, x_scalar, y_scalar)
            }
            Type::Shape(out) => {
                // Shape output - all inputs should be shapes or compatible types
                where_forward_shape(out, &condition, &x, &y)
            }
            Type::Other(_) => panic!("Where cannot handle Other type"),
        }
    }
}

// Helper functions for Where operation
fn where_forward_scalar(
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

fn where_forward_shape(out: &ShapeType, condition: &Type, x: &Type, y: &Type) -> TokenStream {
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

fn where_input_as_tensor(
    input: &Type,
    broadcast_rank: usize,
    scope: &mut Scope,
    node_position: usize,
) -> TokenStream {
    let (tensor, rank) = match input {
        Type::Tensor(t) => {
            let arg_type = onnx_ir::ir::ArgType::Tensor(onnx_ir::ir::TensorType {
                dtype: match t.kind {
                    crate::burn::TensorKind::Float => onnx_ir::ir::DType::F32,
                    crate::burn::TensorKind::Int => onnx_ir::ir::DType::I64,
                    crate::burn::TensorKind::Bool => onnx_ir::ir::DType::Bool,
                },
                rank: t.rank,
                static_shape: None,
            });
            let arg = onnx_ir::Argument::new(t.name.to_string(), arg_type);
            (scope.tensor_use_owned(&arg, node_position), t.rank)
        }
        Type::Scalar(s) => (s.to_full_tensor(&vec![1; broadcast_rank]), broadcast_rank),
        Type::Shape(s) => (s.to_tensor(), 1),
        Type::Other(_) => panic!("Where op: Other input not implemented"),
    };

    if rank < broadcast_rank {
        // Generate unsqueeze_dims to add trailing dimensions for broadcasting
        // Create a vector of dimension indices to unsqueeze at the end
        let dims_to_unsqueeze: Vec<isize> = (rank..broadcast_rank).map(|d| d as isize).collect();
        let dims = quote! { &[#(#dims_to_unsqueeze),*] };
        quote! { #tensor.unsqueeze_dims(#dims) }
    } else {
        tensor
    }
}
