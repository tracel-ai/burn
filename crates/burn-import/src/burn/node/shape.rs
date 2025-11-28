use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::shape::ShapeNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
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
                let input = scope.arg(input_arg);
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

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::ir::{ArgType, Argument, TensorType};
    use onnx_ir::shape::{ShapeConfig, ShapeNode};

    #[test]
    fn test_shape_full() {
        let config = ShapeConfig { start: 0, end: 3 };
        let input = Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::F32, 3, None)),
        );

        let node = ShapeNode {
            name: "shape1".to_string(),
            inputs: vec![input],
            outputs: vec![Argument::new("output", ArgType::Shape(3))],
            config,
        };
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 3>) -> [i64; 3] {
            let output: [i64; 3] = input
                .dims()[0..3]
                .iter()
                .map(|&x| x as i64)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            output
        }
        ");
    }

    #[test]
    fn test_shape_partial() {
        let config = ShapeConfig { start: 1, end: 3 };
        let input = Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::F32, 4, None)),
        );

        let node = ShapeNode {
            name: "shape2".to_string(),
            inputs: vec![input],
            outputs: vec![Argument::new("output", ArgType::Shape(2))],
            config,
        };
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 4>) -> [i64; 2] {
            let output: [i64; 2] = input
                .dims()[1..3]
                .iter()
                .map(|&x| x as i64)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            output
        }
        ");
    }

    #[test]
    fn test_shape_of_shape() {
        let config = ShapeConfig { start: 0, end: 1 };
        let input = Argument::new("input", ArgType::Shape(3));

        let node = ShapeNode {
            name: "shape3".to_string(),
            inputs: vec![input],
            outputs: vec![Argument::new("output", ArgType::Shape(1))],
            config,
        };
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: [i64; 3]) -> [i64; 1] {
            let output: [i64; 1] = [3i64];
            output
        }
        ");
    }
}
