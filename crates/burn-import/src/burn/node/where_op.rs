use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::where_op::WhereNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let condition_arg = &self.inputs[0];
        let x_arg = &self.inputs[1];
        let y_arg = &self.inputs[2];
        let output_arg = self.outputs.first().unwrap();
        let output = arg_to_ident(output_arg);

        match &output_arg.ty {
            ArgType::Tensor(out_tensor) => {
                let broadcast_rank = out_tensor.rank;

                // Get condition as tensor
                let cond = where_input_as_tensor(condition_arg, broadcast_rank, scope);

                // Get y as tensor
                let y_tensor = where_input_as_tensor(y_arg, broadcast_rank, scope);

                // Check if x is a scalar - if so, use mask_fill
                if let ArgType::Scalar(_) = &x_arg.ty {
                    let x_name = arg_to_ident(x_arg);
                    quote! {
                        let #output = #y_tensor.mask_fill(#cond, #x_name);
                    }
                } else {
                    // x is tensor or shape - use mask_where
                    let x_tensor = where_input_as_tensor(x_arg, broadcast_rank, scope);
                    quote! {
                        let #output = #y_tensor.mask_where(#cond, #x_tensor);
                    }
                }
            }
            ArgType::Scalar(_) => {
                // Scalar output means all inputs are scalars
                let cond_name = arg_to_ident(condition_arg);
                let x_name = arg_to_ident(x_arg);
                let y_name = arg_to_ident(y_arg);

                quote! {
                    let #output = if #cond_name {
                        #x_name
                    } else {
                        #y_name
                    };
                }
            }
            ArgType::Shape(_) => {
                // Shape output - handle element-wise or whole shape selection
                match (&condition_arg.ty, &x_arg.ty, &y_arg.ty) {
                    (ArgType::Shape(_), ArgType::Shape(_), ArgType::Shape(_)) => {
                        // Element-wise selection between shape dimensions
                        let cond_name = arg_to_ident(condition_arg);
                        let x_name = arg_to_ident(x_arg);
                        let y_name = arg_to_ident(y_arg);

                        quote! {
                            let #output = {
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
                    (ArgType::Scalar(_), ArgType::Shape(_), ArgType::Shape(_)) => {
                        // Scalar condition: select entire shape x or y
                        let cond_name = arg_to_ident(condition_arg);
                        let x_name = arg_to_ident(x_arg);
                        let y_name = arg_to_ident(y_arg);

                        quote! {
                            let #output = if #cond_name { #x_name } else { #y_name };
                        }
                    }
                    _ => panic!(
                        "Where with Shape output only supports: \
                         (Shape, Shape, Shape) for element-wise selection or \
                         (Scalar, Shape, Shape) for whole shape selection"
                    ),
                }
            }
        }
    }
}

// Helper function to convert an input to a tensor for broadcasting
fn where_input_as_tensor(
    arg: &Argument,
    broadcast_rank: usize,
    scope: &mut super::super::scope::ScopeAtPosition<'_>,
) -> TokenStream {
    match &arg.ty {
        ArgType::Tensor(t) => {
            let tensor = scope.arg(arg);
            let rank = t.rank;

            if rank < broadcast_rank {
                // Unsqueeze to match broadcast rank
                let dims_to_unsqueeze: Vec<isize> =
                    (rank..broadcast_rank).map(|d| d as isize).collect();
                quote! { #tensor.unsqueeze_dims(&[#(#dims_to_unsqueeze),*]) }
            } else {
                tensor
            }
        }
        ArgType::Scalar(_) => {
            // Convert scalar to full tensor with broadcast_rank dimensions
            let name = arg_to_ident(arg);
            let shape_vec: Vec<_> = (0..broadcast_rank).map(|_| quote! { 1 }).collect();
            quote! {
                Tensor::from_data([[#(#shape_vec),*]; 1], &*self.device).mul_scalar(#name)
            }
        }
        ArgType::Shape(_) => {
            // Convert shape to tensor (rank 1)
            let name = arg_to_ident(arg);
            let tensor = quote! {
                Tensor::<B, 1, burn::tensor::Int>::from_data(&#name as &[_], &*self.device)
            };

            if broadcast_rank > 1 {
                // Unsqueeze to match broadcast rank
                let dims_to_unsqueeze: Vec<isize> =
                    (1..broadcast_rank).map(|d| d as isize).collect();
                quote! { #tensor.unsqueeze_dims(&[#(#dims_to_unsqueeze),*]) }
            } else {
                tensor
            }
        }
    }
}
