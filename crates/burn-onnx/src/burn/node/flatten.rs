use super::prelude::*;

impl NodeCodegen for onnx_ir::flatten::FlattenNode {
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

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::flatten::{FlattenConfig, FlattenNode, FlattenNodeBuilder};

    fn create_flatten_node(name: &str, axis: usize) -> FlattenNode {
        let config = FlattenConfig::new(axis);

        FlattenNodeBuilder::new(name)
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_flatten_forward_axis_0() {
        let node = create_flatten_node("flatten1", 0);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
            let output = input.reshape::<2>([1, -1]);
            output
        }
        ");
    }

    #[test]
    fn test_flatten_forward_axis_1() {
        let node = create_flatten_node("flatten1", 1);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
            let output = {
                let leading_dim = input.shape().dims[..1].iter().product::<usize>() as i32;
                input.reshape::<2, _>([leading_dim, -1])
            };
            output
        }
        ");
    }

    #[test]
    fn test_flatten_forward_axis_2() {
        let node = create_flatten_node("flatten1", 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
            let output = {
                let leading_dim = input.shape().dims[..2].iter().product::<usize>() as i32;
                input.reshape::<2, _>([leading_dim, -1])
            };
            output
        }
        ");
    }
}
