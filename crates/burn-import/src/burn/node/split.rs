use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::split::SplitNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let axis = self.config.axis.to_tokens();

        let outputs = self.outputs.iter().map(arg_to_ident).collect::<Vec<_>>();

        let unpack_outputs = quote! {
            let [#(#outputs),*] = split_tensors.try_into().unwrap();
        };

        if let Some(split_sizes_input) = &self.config.split_sizes {
            // Extract static split sizes from the enum wrapper
            let split_sizes = match split_sizes_input {
                onnx_ir::split::SplitSizesInput::Static(sizes) => sizes,
                onnx_ir::split::SplitSizesInput::Runtime(_) => {
                    panic!("Runtime split sizes are not supported in burn-import")
                }
            };
            let split_sizes_tokens = split_sizes.iter().map(|s| s.to_tokens());
            quote! {
                let split_tensors = #input.split_with_sizes(vec![#(#split_sizes_tokens),*], #axis);
                #unpack_outputs
            }
        } else {
            let split_size = &self.config.split_size.unwrap();
            let split_size_tokens = split_size.to_tokens();
            quote! {
                let split_tensors = #input.split(#split_size_tokens, #axis);
                #unpack_outputs
            }
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        // When split_sizes is used, we generate vec![...] which needs the vec macro
        if self.config.split_sizes.is_some() {
            imports.register("alloc::vec");
        }
    }
}
