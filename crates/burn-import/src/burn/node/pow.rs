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
