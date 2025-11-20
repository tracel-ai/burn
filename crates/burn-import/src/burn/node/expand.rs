use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::expand::ExpandNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        let output_rank = match &self.outputs.first().unwrap().ty {
            ArgType::Tensor(tensor) => tensor.rank,
            _ => panic!("Expand output must be a tensor"),
        };

        let shape = match &self.config {
            onnx_ir::expand::ExpandConfig::Static(static_shape) => static_shape.to_tokens(),
            onnx_ir::expand::ExpandConfig::Runtime(shape_ref) => {
                // Get the actual argument using the RuntimeInputRef
                let shape_arg = &self.inputs[shape_ref.input_index];
                match &shape_arg.ty {
                    ArgType::Tensor(_) => {
                        let tensor_name = arg_to_ident(shape_arg);
                        quote! {
                            TryInto::<[B::IntElem; #output_rank]>::try_into(#tensor_name.to_data().as_slice::<B::IntElem>().unwrap()).unwrap()
                        }
                    }
                    ArgType::Shape(_) => {
                        // Shape arrays are [i64; N] and expand now accepts them directly via Element trait
                        let shape_name = arg_to_ident(shape_arg);
                        quote! { #shape_name }
                    }
                    _ => panic!("Invalid shape source {:?}", shape_arg.ty),
                }
            }
        };

        quote! {
            let #output = #input.expand(#shape);
        }
    }
}
