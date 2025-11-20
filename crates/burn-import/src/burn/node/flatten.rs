use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::flatten::FlattenNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        if self.config.axis == 0 {
            quote! {
                let #output = #input.reshape::<2>([1, -1]);
            }
        } else {
            let axis = self.config.axis.to_tokens();
            quote! {
                let #output = {
                    let leading_dim = #input.shape().dims[..#axis].iter().product::<usize>() as i32;
                    #input.reshape::<2, _>([leading_dim, -1])
                };
            }
        }
    }
}
