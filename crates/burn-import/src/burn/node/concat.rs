use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::concat::ConcatNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let output = arg_to_ident(self.outputs.first().unwrap());
        let dim = self.config.axis.to_tokens();

        // Determine if this is tensor or shape concatenation based on output type
        match &self.outputs.first().unwrap().ty {
            ArgType::Tensor(_) => {
                // Tensor concatenation
                let inputs = self.inputs.iter().map(|arg| scope.arg(arg));

                quote! {
                    let #output = burn::tensor::Tensor::cat([#(#inputs),*].into(), #dim);
                }
            }
            ArgType::Shape(shape) => {
                // Shape concatenation - shapes are 1D so concat is always on axis 0
                if self.config.axis != 0 {
                    panic!(
                        "Shape concatenation only supports dim=0, got dim={}",
                        self.config.axis
                    );
                }
                let output_rank = shape;

                // Generate code to concatenate shape arrays
                let mut shape_parts = Vec::new();
                for input in &self.inputs {
                    let input_name = arg_to_ident(input);
                    shape_parts.push(quote! { &#input_name[..] });
                }

                quote! {
                    let #output: [i64; #output_rank] = [#(#shape_parts),*].concat().try_into().unwrap();
                }
            }
            _ => panic!("Concat only supports Tensor or Shape outputs"),
        }
    }
}
