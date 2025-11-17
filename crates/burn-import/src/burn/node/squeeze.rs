use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, ToTokens, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct SqueezeNode {
    pub input: Type,
    pub output: Type,
    pub axes: Option<Vec<i64>>,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for SqueezeNode {
    fn output_types(&self) -> Vec<Type> {
        vec![self.output.clone()]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![self.input.clone()]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        match (&self.input, &self.output) {
            (Type::Tensor(input), Type::Tensor(output)) => {
                let input_tensor = scope.tensor_use_owned(input, node_position);
                let output_name = &output.name;

                match &self.axes {
                    Some(axes_vec) => {
                        // Use squeeze_dims with specific axes
                        let axes_arg = axes_vec.to_tokens();
                        quote! {
                            let #output_name = #input_tensor.squeeze_dims(&#axes_arg);
                        }
                    }
                    None => {
                        // When axes is None, squeeze all dimensions with size 1
                        let output_rank = output.rank;
                        quote! {
                            let #output_name = #input_tensor.squeeze::<#output_rank>();
                        }
                    }
                }
            }
            (Type::Shape(input), Type::Scalar(output)) => {
                // Shape(1) squeezed on axis 0 produces a scalar
                let input_name = &input.name;
                let output_name = &output.name;

                // Cast to the appropriate scalar type
                let cast_expr = match &output.kind {
                    crate::burn::ScalarKind::Int64 => quote! { #input_name[0] as i64 },
                    crate::burn::ScalarKind::Int32 => quote! { #input_name[0] as i32 },
                    _ => panic!(
                        "Squeeze from Shape to Scalar only supports Int32/Int64 output types"
                    ),
                };

                quote! {
                    let #output_name = #cast_expr;
                }
            }
            (Type::Shape(input), Type::Shape(output)) => {
                // Shape(n) where n > 1 remains unchanged (squeeze is a no-op)
                let input_name = &input.name;
                let output_name = &output.name;

                quote! {
                    let #output_name = #input_name;
                }
            }
            (Type::Scalar(input), Type::Scalar(output)) => {
                // Scalar squeeze is a no-op
                let input_name = &input.name;
                let output_name = &output.name;

                quote! {
                    let #output_name = #input_name;
                }
            }
            (Type::Tensor(input), Type::Scalar(output)) => {
                // This handles ONNX models where single-element tensors need to be converted to scalars
                // Works for all tensor types (Float, Int, Bool) using the .into_scalar() method
                let input = scope.tensor_use_owned(input, node_position);
                let output_name = &output.name;

                // Use .into_scalar() and cast to the appropriate concrete type using .elem::<T>()
                let elem_cast = match &output.kind {
                    crate::burn::ScalarKind::Float32 => quote! { .elem::<f32>() },
                    crate::burn::ScalarKind::Float64 => quote! { .elem::<f64>() },
                    crate::burn::ScalarKind::Int32 => quote! { .elem::<i32>() },
                    crate::burn::ScalarKind::Int64 => quote! { .elem::<i64>() },
                    crate::burn::ScalarKind::Bool => quote! { .elem::<bool>() },
                };

                quote! {
                    let #output_name = #input.into_scalar()#elem_cast;
                }
            }
            _ => panic!(
                "Squeeze: unsupported input/output combination: {:?} -> {:?}",
                self.input, self.output
            ),
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Squeeze(self)
    }
}

impl OnnxIntoNode for SqueezeNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::Squeeze(n) = &node else {
            panic!("Expected Squeeze node");
        };
        let inputs = &n.inputs;
        let outputs = &n.outputs;
        let config = &n.config;
        let input = Type::from(inputs.first().unwrap());
        let output = Type::from(outputs.first().unwrap());
        let axes = config.axes.as_ref().map(|a| match a {
            onnx_ir::node::squeeze::SqueezeInput::Static(axes) => axes.clone(),
            onnx_ir::node::squeeze::SqueezeInput::Runtime(_) => {
                panic!("Runtime squeeze axes not yet supported in burn-import")
            }
        });
        Self::new(input, output, axes)
    }
}
