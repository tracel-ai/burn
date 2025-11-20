use super::prelude::*;
use std::str::FromStr;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::pad::PadNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        // Extract static pads from the enum wrapper
        let pads_vec = match &self.config.pads {
            onnx_ir::pad::PadInput::Static(pads) => pads,
            onnx_ir::pad::PadInput::Runtime(_) => {
                panic!("Runtime pads are not supported in burn-import")
            }
        };
        let pads = pads_vec.iter().map(|p| p.to_tokens());

        // Extract static constant value from the enum wrapper
        let constant_value_f32 = match &self.config.constant_value {
            onnx_ir::pad::ConstantValueInput::Static(value) => value,
            onnx_ir::pad::ConstantValueInput::Runtime(_) => {
                panic!("Runtime constant value is not supported in burn-import")
            }
        };
        let constant_value_string = format!("{}_f32", constant_value_f32);
        let constant_value = TokenStream::from_str(&constant_value_string).unwrap();

        quote! {
            let #output = #input.pad((#(#pads),*), #constant_value);
        }
    }
}
