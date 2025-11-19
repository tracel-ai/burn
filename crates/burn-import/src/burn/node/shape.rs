use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::shape::ShapeNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        use onnx_ir::ir::ArgType;

        let input_arg = self.inputs.first().unwrap();
        let output_arg = self.outputs.first().unwrap();
        let output = arg_to_ident(output_arg);

        let dim = match &output_arg.ty {
            ArgType::Shape(rank) => rank.to_tokens(),
            _ => panic!("Shape operation expects Shape output"),
        };

        let start_dim_tok = self.config.start.to_tokens();
        let end_dim_tok = self.config.end.to_tokens();

        let function = match &input_arg.ty {
            ArgType::Tensor(_) => {
                let input = scope.tensor_use_owned(input_arg, node_position);
                quote! {
                    #input.dims()[#start_dim_tok..#end_dim_tok]
                        .iter()
                        .map(|&x| x as i64)
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap()
                }
            }
            ArgType::Shape(shape_rank) => {
                // If input is already a shape array [i64; N], the Shape operation
                // returns the dimensionality of the shape (which is N) as a Shape(1) array
                // This matches the ONNX semantics where Shape of a shape gives you the rank
                let rank_value = *shape_rank as i64;
                quote! { [#rank_value] }
            }
            ArgType::Scalar(_) => panic!("Shape operation only supports Tensor or Shape inputs"),
        };

        quote! {
            let #output: [i64;#dim] = #function;
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        // Only register Vec if we're extracting shape from a tensor
        if self.inputs.first().unwrap().ty.is_tensor() {
            imports.register("alloc::vec::Vec");
        }
    }
}
