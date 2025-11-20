use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::clip::ClipNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        // Extract static values from ClipInput enum
        let min = match &self.config.min {
            Some(onnx_ir::node::clip::ClipInput::Static(v)) => Some(*v),
            Some(onnx_ir::node::clip::ClipInput::Runtime(_)) => {
                panic!("Clip: runtime min values are not supported in burn-import")
            }
            None => None,
        };
        let max = match &self.config.max {
            Some(onnx_ir::node::clip::ClipInput::Static(v)) => Some(*v),
            Some(onnx_ir::node::clip::ClipInput::Runtime(_)) => {
                panic!("Clip: runtime max values are not supported in burn-import")
            }
            None => None,
        };

        if let Some(min) = min {
            if let Some(max) = max {
                quote! {
                    let #output = #input.clamp(#min, #max);
                }
            } else {
                quote! {
                    let #output = #input.clamp_min(#min);
                }
            }
        } else if let Some(max) = max {
            quote! {
                let #output = #input.clamp_max(#max);
            }
        } else {
            panic!("Clip node must have at least one min or max value");
        }
    }
}
