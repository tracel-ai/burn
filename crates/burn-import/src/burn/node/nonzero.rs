use super::prelude::*;
use crate::burn::TensorKind;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::nonzero::NonZeroNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        // Determine input tensor kind
        let input_arg = self.inputs.first().unwrap();
        let input_kind = match &input_arg.ty {
            ArgType::Tensor(t) => TensorKind::from(t.dtype),
            _ => panic!("Expected tensor input"),
        };

        // Generate the appropriate zero value based on input tensor type
        match input_kind {
            TensorKind::Float => {
                quote! {
                    let #output = #input.not_equal_elem(0.0).argwhere().transpose();
                }
            }
            TensorKind::Int => {
                quote! {
                    let #output = #input.not_equal_elem(0).argwhere().transpose();
                }
            }
            TensorKind::Bool => {
                // For bool tensors, we can use argwhere directly since false is the "zero" value
                // ONNX NonZero expects output shape [rank, num_nonzero] but argwhere returns [num_nonzero, rank]
                // So we need to transpose the result
                quote! {
                    let #output = #input.argwhere().transpose();
                }
            }
        }
    }
}
