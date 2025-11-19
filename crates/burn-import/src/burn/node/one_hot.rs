use super::prelude::*;
use crate::burn::TensorKind;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::one_hot::OneHotNode {
    fn inputs(&self) -> &[Argument] {
        // Only the first input (indices) is a dynamic tensor
        // depth and values are either static or runtime inputs
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(self.inputs.first().unwrap(), node_position);
        let output = arg_to_ident(self.outputs.first().unwrap());

        // Extract num_classes from config.depth
        let num_classes = match &self.config.depth {
            onnx_ir::one_hot::OneHotDepthInput::Static(d) => quote! { #d },
            onnx_ir::one_hot::OneHotDepthInput::Runtime(_) => {
                panic!("OneHot with runtime depth is not supported in burn-import")
            }
        };

        // Extract values from config.values
        let (on_value, off_value) = match &self.config.values {
            onnx_ir::one_hot::OneHotValuesInput::Static(v) => {
                let off = v[0];
                let on = v[1];
                (quote! { #on }, quote! { #off })
            }
            onnx_ir::one_hot::OneHotValuesInput::Runtime(_) => {
                panic!("OneHot with runtime values is not supported in burn-import")
            }
        };

        let axis = self.config.axis;

        // Determine input and output tensor kinds
        let input_arg = self.inputs.first().unwrap();
        let output_arg = self.outputs.first().unwrap();

        let input_kind = match &input_arg.ty {
            ArgType::Tensor(t) => TensorKind::from(t.dtype),
            _ => panic!("Expected tensor input"),
        };

        let output_kind = match &output_arg.ty {
            ArgType::Tensor(t) => TensorKind::from(t.dtype),
            _ => panic!("Expected tensor output"),
        };

        match (input_kind, output_kind) {
            (TensorKind::Int, TensorKind::Int) | (TensorKind::Float, TensorKind::Float) => {
                quote! {
                    let #output = #input.one_hot_fill(#num_classes, #on_value, #off_value, #axis);
                }
            }
            (TensorKind::Int, TensorKind::Float) => {
                quote! {
                    let #output = #input.one_hot_fill(#num_classes, #on_value, #off_value, #axis).float();
                }
            }
            (TensorKind::Float, TensorKind::Int) => {
                quote! {
                    let #output = #input.one_hot_fill(#num_classes, #on_value, #off_value, #axis).int();
                }
            }
            (TensorKind::Int, TensorKind::Bool) | (TensorKind::Float, TensorKind::Bool) => {
                quote! {
                    let #output = #input.one_hot_fill(#num_classes, #on_value, #off_value, #axis).bool();
                }
            }
            (TensorKind::Bool, _) => panic!("Input should be numeric"),
        }
    }
}
