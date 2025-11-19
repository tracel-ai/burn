use super::{NodeCodegen, arg_to_ident};
use crate::burn::{Scope, ToTokens};
use burn::record::PrecisionSettings;
use onnx_ir::Argument;
use proc_macro2::TokenStream;
use quote::quote;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::squeeze::SqueezeNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        use onnx_ir::ir::ArgType;

        let input_arg = self.inputs.first().unwrap();
        let output_arg = self.outputs.first().unwrap();
        let output = arg_to_ident(output_arg);

        match (&input_arg.ty, &output_arg.ty) {
            (ArgType::Tensor(_), ArgType::Tensor(output_tensor)) => {
                let input = scope.tensor_use_owned(input_arg, node_position);

                match &self.config.axes {
                    Some(onnx_ir::squeeze::SqueezeInput::Static(axes_vec)) => {
                        // Use squeeze_dims with specific axes
                        let axes_arg = axes_vec.to_tokens();
                        let output_rank = output_tensor.rank.to_tokens();
                        quote! {
                            let #output = #input.squeeze_dims::<#output_rank>(&#axes_arg);
                        }
                    }
                    Some(onnx_ir::squeeze::SqueezeInput::Runtime(_)) => {
                        panic!("Runtime squeeze axes not yet supported in burn-import")
                    }
                    None => {
                        // When axes is None, squeeze all dimensions with size 1
                        let output_rank = output_tensor.rank.to_tokens();
                        quote! {
                            let #output = #input.squeeze::<#output_rank>();
                        }
                    }
                }
            }
            (ArgType::Shape(_), ArgType::Scalar(elem_type)) => {
                // Shape(1) squeezed on axis 0 produces a scalar
                let input_name = arg_to_ident(input_arg);

                use onnx_ir::ir::DType;
                let cast_expr = match elem_type {
                    DType::I64 => quote! { #input_name[0] as i64 },
                    DType::I32 => quote! { #input_name[0] as i32 },
                    _ => panic!(
                        "Squeeze from Shape to Scalar only supports Int32/Int64 output types"
                    ),
                };

                quote! {
                    let #output = #cast_expr;
                }
            }
            (ArgType::Shape(_), ArgType::Shape(_)) => {
                // Shape(n) where n > 1 remains unchanged (squeeze is a no-op)
                let input_name = arg_to_ident(input_arg);

                quote! {
                    let #output = #input_name;
                }
            }
            (ArgType::Scalar(_), ArgType::Scalar(_)) => {
                // Scalar squeeze is a no-op
                let input_name = arg_to_ident(input_arg);

                quote! {
                    let #output = #input_name;
                }
            }
            (ArgType::Tensor(_), ArgType::Scalar(elem_type)) => {
                // Single-element tensor to scalar conversion
                let input = scope.tensor_use_owned(input_arg, node_position);

                use onnx_ir::ir::DType;
                let elem_cast = match elem_type {
                    DType::F32 => quote! { .elem::<f32>() },
                    DType::F64 => quote! { .elem::<f64>() },
                    DType::I32 => quote! { .elem::<i32>() },
                    DType::I64 => quote! { .elem::<i64>() },
                    DType::Bool => quote! { .elem::<bool>() },
                    _ => panic!("Unsupported scalar type: {:?}", elem_type),
                };

                quote! {
                    let #output = #input.into_scalar()#elem_cast;
                }
            }
            _ => panic!(
                "Squeeze: unsupported input/output combination: {:?} -> {:?}",
                input_arg.ty, output_arg.ty
            ),
        }
    }
}
