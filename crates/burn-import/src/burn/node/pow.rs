use super::prelude::*;
use onnx_ir::DType;

/// Type of power operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerType {
    /// Integer power (powi)
    Int,
    /// Float power (powf)
    Float,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::pow::PowNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let lhs_arg = self.inputs.first().unwrap();
        let rhs_arg = self.inputs.get(1).unwrap();
        let output = arg_to_ident(self.outputs.first().unwrap());

        let lhs = scope.arg(lhs_arg);

        let rhs = scope.arg(rhs_arg);

        // Determine power type based on RHS type
        let power_type = match &rhs_arg.ty {
            ArgType::Tensor(t) => match t.dtype {
                DType::I64 | DType::I32 | DType::I16 | DType::I8 => PowerType::Int,
                DType::F64 | DType::F32 | DType::F16 | DType::BF16 | DType::Flex32 => {
                    PowerType::Float
                }
                _ => panic!("pow function requires RHS to be int or float type"),
            },
            ArgType::Scalar(dtype) => match dtype {
                DType::I64 | DType::I32 | DType::I16 | DType::I8 => PowerType::Int,
                DType::F64 | DType::F32 | DType::F16 | DType::BF16 | DType::Flex32 => {
                    PowerType::Float
                }
                _ => panic!("pow function requires RHS to be int or float type"),
            },
            _ => panic!("pow function only supports RHS scalar or tensor types"),
        };

        let function = match (power_type, &rhs_arg.ty) {
            (PowerType::Int, ArgType::Tensor(_)) => quote! { #lhs.powi(#rhs) },
            (PowerType::Int, ArgType::Scalar(_)) => quote! { #lhs.powi_scalar(#rhs) },
            (PowerType::Float, ArgType::Tensor(_)) => quote! { #lhs.powf(#rhs) },
            (PowerType::Float, ArgType::Scalar(_)) => quote! { #lhs.powf_scalar(#rhs) },
            _ => panic!("Invalid power type combination"),
        };

        quote! {
            let #output = #function;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::pow::{PowNode, PowNodeBuilder};

    fn create_pow_node_tensor_tensor(name: &str, base_dtype: DType, exp_dtype: DType) -> PowNode {
        PowNodeBuilder::new(name)
            .input_tensor("base", 2, base_dtype)
            .input_tensor("exponent", 2, exp_dtype)
            .output_tensor("output", 2, base_dtype)
            .build()
    }

    fn create_pow_node_tensor_scalar(name: &str, base_dtype: DType, exp_dtype: DType) -> PowNode {
        PowNodeBuilder::new(name)
            .input_tensor("base", 2, base_dtype)
            .input_scalar("exponent", exp_dtype)
            .output_tensor("output", 2, base_dtype)
            .build()
    }

    #[test]
    fn test_pow_float_tensor_float_tensor() {
        let node = create_pow_node_tensor_tensor("pow1", DType::F32, DType::F32);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let output = base.powf(exponent);");
    }

    #[test]
    fn test_pow_float_tensor_int_tensor() {
        let node = create_pow_node_tensor_tensor("pow1", DType::F32, DType::I32);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let output = base.powi(exponent);");
    }

    #[test]
    fn test_pow_float_tensor_int_scalar() {
        let node = create_pow_node_tensor_scalar("pow1", DType::F32, DType::I32);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let output = base.powi_scalar(exponent);");
    }

    #[test]
    fn test_pow_float_tensor_float_scalar() {
        let node = create_pow_node_tensor_scalar("pow1", DType::F32, DType::F32);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let output = base.powf_scalar(exponent);");
    }

    #[test]
    fn test_pow_int_tensor_int_scalar() {
        let node = create_pow_node_tensor_scalar("pow1", DType::I32, DType::I32);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let output = base.powi_scalar(exponent);");
    }
}
