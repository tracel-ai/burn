use super::{NodeCodegen, arg_to_ident};
use crate::burn::{BurnImports, Scope, ToTokens};
use burn::record::PrecisionSettings;
use onnx_ir::Argument;
use proc_macro2::TokenStream;
use quote::quote;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::unsqueeze::UnsqueezeNode {
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
        use onnx_ir::ir::ArgType;

        let input_arg = self.inputs.first().unwrap();
        let output_arg = self.outputs.first().unwrap();
        let output = arg_to_ident(output_arg);

        // Generate axes token stream
        let axes = match &self.config {
            onnx_ir::unsqueeze::UnsqueezeConfig::Static(static_axes) => static_axes.to_tokens(),
            onnx_ir::unsqueeze::UnsqueezeConfig::Runtime(axes_ref) => {
                let axes_arg = &self.inputs[axes_ref.input_index];
                match &axes_arg.ty {
                    ArgType::Tensor(_) => {
                        let tensor_name = arg_to_ident(axes_arg);
                        quote! {
                            #tensor_name.to_data().as_slice::<B::IntElem>().unwrap().iter().map(|&x| x.to_isize()).collect::<Vec<isize>>()
                        }
                    }
                    _ => panic!(
                        "UnsqueezeNode received invalid axes type: expected tensor but got {:?}",
                        axes_arg.ty
                    ),
                }
            }
        };

        match (&input_arg.ty, &output_arg.ty) {
            (ArgType::Tensor(_input_tensor), ArgType::Tensor(output_tensor)) => {
                let input = scope.tensor_use_owned(input_arg, node_position);
                let output_rank = output_tensor.rank.to_tokens();

                // Generate the correct output type based on the tensor kind
                let output_type = match &output_tensor.dtype {
                    onnx_ir::ir::DType::I8
                    | onnx_ir::ir::DType::I32
                    | onnx_ir::ir::DType::I64
                    | onnx_ir::ir::DType::U8 => {
                        quote! { Tensor<B, #output_rank, Int> }
                    }
                    onnx_ir::ir::DType::F32 | onnx_ir::ir::DType::F64 => {
                        quote! { Tensor<B, #output_rank> }
                    }
                    onnx_ir::ir::DType::Bool => {
                        quote! { Tensor<B, #output_rank, Bool> }
                    }
                    _ => panic!("Unsupported tensor dtype: {:?}", output_tensor.dtype),
                };

                quote! {
                    let #output: #output_type = #input.unsqueeze_dims::<#output_rank>(&#axes);
                }
            }
            (ArgType::Scalar(_scalar_type), ArgType::Tensor(output_tensor)) => {
                let scalar_name = arg_to_ident(input_arg);
                let output_rank = output_tensor.rank.to_tokens();

                // Determine the element type based on the output tensor type
                let tensor_creation = match &output_tensor.dtype {
                    onnx_ir::ir::DType::I8
                    | onnx_ir::ir::DType::I32
                    | onnx_ir::ir::DType::I64
                    | onnx_ir::ir::DType::U8 => {
                        let elem_conversion = quote! { #scalar_name.elem::<B::IntElem>() };
                        quote! { Tensor::<B, #output_rank, Int>::from_data([#elem_conversion], &self.device).unsqueeze() }
                    }
                    onnx_ir::ir::DType::F32 | onnx_ir::ir::DType::F64 => {
                        let elem_conversion = quote! { #scalar_name.elem::<B::FloatElem>() };
                        quote! { Tensor::<B, #output_rank>::from_data([#elem_conversion], &self.device).unsqueeze() }
                    }
                    onnx_ir::ir::DType::Bool => {
                        let elem_conversion = quote! { #scalar_name != 0 };
                        quote! { Tensor::<B, #output_rank, Bool>::from_data([#elem_conversion], &self.device).unsqueeze() }
                    }
                    _ => panic!("Unsupported tensor dtype: {:?}", output_tensor.dtype),
                };

                quote! {
                    let #output = #tensor_creation;
                }
            }
            (ArgType::Scalar(scalar_type), ArgType::Shape(_)) => {
                // Scalar(Int) -> Shape[1] conversion
                let input_name = arg_to_ident(input_arg);

                use onnx_ir::ir::DType;
                let value_expr = match scalar_type {
                    DType::I64 => quote! { #input_name },
                    DType::I32 => quote! { #input_name as i64 },
                    _ => panic!(
                        "Unsqueeze from Scalar to Shape only supports Int32/Int64 input types, but got: {:?}",
                        scalar_type
                    ),
                };

                quote! {
                    let #output = [#value_expr];
                }
            }
            _ => panic!(
                "UnsqueezeNode received unsupported input/output combination: {:?} -> {:?}",
                input_arg.ty, output_arg.ty
            ),
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        match &self.config {
            onnx_ir::unsqueeze::UnsqueezeConfig::Runtime(_) => {
                imports.register("alloc::vec::Vec");
            }
            _ => {}
        }
    }
}
