use super::prelude::*;
use onnx_ir::ir::ArgType;

impl<PS: PrecisionSettings> NodeCodegen<PS>
    for onnx_ir::node::constant_of_shape::ConstantOfShapeNode
{
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, _scope: &mut Scope, _node_position: usize) -> TokenStream {
        let output = arg_to_ident(self.outputs.first().unwrap());

        // Extract fill value from config
        let value = if let Some(tensor_data) = &self.config.value {
            // Extract the scalar value from the tensor data
            match tensor_data.dtype {
                onnx_ir::ir::DType::F32 => {
                    let val = tensor_data.as_slice::<f32>().unwrap()[0];
                    quote! { #val }
                }
                onnx_ir::ir::DType::F64 => {
                    let val = tensor_data.as_slice::<f64>().unwrap()[0];
                    quote! { #val }
                }
                onnx_ir::ir::DType::I32 => {
                    let val = tensor_data.as_slice::<i32>().unwrap()[0];
                    quote! { #val }
                }
                onnx_ir::ir::DType::I64 => {
                    let val = tensor_data.as_slice::<i64>().unwrap()[0];
                    quote! { #val }
                }
                onnx_ir::ir::DType::Bool => {
                    let val = tensor_data.as_slice::<bool>().unwrap()[0];
                    quote! { #val }
                }
                _ => quote! { 0.0f32 }, // Default fallback
            }
        } else {
            quote! { 0.0f32 } // Default value per ONNX spec
        };

        // Generate shape expression based on Static or Runtime
        let shape_expr = match &self.config.shape {
            onnx_ir::node::constant_of_shape::ConstantOfShapeShape::Static(static_shape) => {
                // Static shape values - embed them directly in the code
                let shape_values = static_shape.iter().map(|v| {
                    let val = *v as usize;
                    quote! { #val }
                });
                quote! { [#(#shape_values),*] }
            }
            onnx_ir::node::constant_of_shape::ConstantOfShapeShape::Runtime(runtime_ref) => {
                // Runtime shape input
                let arg = &self.inputs[runtime_ref.input_index];
                let input_name = arg_to_ident(arg);
                quote! { #input_name }
            }
        };

        // Generate code based on output type
        match &self.outputs[0].ty {
            ArgType::Scalar(_) => {
                // For scalar output, the input shape should be empty (rank 0)
                // Just return the constant value directly
                quote! {
                    let #output = #value;
                }
            }
            ArgType::Tensor(tensor) => {
                let output_rank = tensor.rank.to_tokens();

                // Check if value is boolean - special handling needed
                let is_bool_value = if let Some(tensor_data) = &self.config.value {
                    matches!(tensor_data.dtype, onnx_ir::ir::DType::Bool)
                } else {
                    false
                };

                if is_bool_value {
                    // Boolean tensors need special handling
                    let bool_val = if let Some(tensor_data) = &self.config.value {
                        tensor_data.as_slice::<bool>().unwrap()[0]
                    } else {
                        false
                    };

                    if bool_val {
                        quote! {
                            let #output = Tensor::<B, #output_rank, Int>::ones(#shape_expr, &*self.device).bool();
                        }
                    } else {
                        quote! {
                            let #output = Tensor::<B, #output_rank, Int>::zeros(#shape_expr, &*self.device).bool();
                        }
                    }
                } else {
                    quote! {
                        let #output = Tensor::full(#shape_expr, #value, &*self.device);
                    }
                }
            }
            ArgType::Shape(_) => {
                // Optimization: When ConstantOfShape outputs Shape(1) with Int64,
                // we directly create a shape array instead of a tensor
                quote! {
                    let #output: [i64; 1] = [#value];
                }
            }
        }
    }
}
