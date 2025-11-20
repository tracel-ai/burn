use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::is_inf::IsInfNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input_arg = self.inputs.first().unwrap();
        let output_arg = self.outputs.first().unwrap();

        let input = scope.arg(input_arg);
        let output = arg_to_ident(output_arg);

        let function = match &output_arg.ty {
            ArgType::Scalar(_) => {
                match (self.config.detect_negative, self.config.detect_positive) {
                    (true, true) => quote! { #input.is_infinite() },
                    (false, true) => quote! { #input.is_infinite() && #input.is_sign_positive() },
                    (true, false) => quote! { #input.is_infinite() && #input.is_sign_negative() },
                    (false, false) => quote! { false },
                }
            }
            ArgType::Tensor(_) => {
                match (self.config.detect_negative, self.config.detect_positive) {
                    (true, true) => quote! { #input.is_inf() },
                    (false, true) => {
                        quote! { #input.clone().is_inf().bool_and(#input.greater_elem(0.0)) }
                    }
                    (true, false) => {
                        quote! { #input.clone().is_inf().bool_and(#input.lower_elem(0.0)) }
                    }
                    (false, false) => quote! { #input.zeros_like().bool() },
                }
            }
            _ => panic!("IsInf only supports scalar or tensor outputs"),
        };

        quote! {
            let #output = #function;
        }
    }

    fn register_imports(&self, _imports: &mut BurnImports) {
        // No special imports needed - is_inf() is a tensor method
    }
}
