use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::{Literal, TokenStream};
use quote::quote;

/// Range parameter that can be either static or runtime
#[derive(Debug, Clone)]
pub enum RangeParam {
    Static(i64),
    Runtime(Type),
}

#[derive(Debug, Clone)]
pub struct RangeNode {
    pub start: RangeParam,
    pub limit: RangeParam,
    pub delta: RangeParam,
    pub output: TensorType,
}

impl RangeNode {
    pub fn new(
        start: RangeParam,
        limit: RangeParam,
        delta: RangeParam,
        output: TensorType,
    ) -> Self {
        Self {
            start,
            limit,
            delta,
            output,
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for RangeNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<Type> {
        let mut inputs = Vec::new();

        if let RangeParam::Runtime(ref t) = self.start {
            inputs.push(t.clone());
        }
        if let RangeParam::Runtime(ref t) = self.limit {
            inputs.push(t.clone());
        }
        if let RangeParam::Runtime(ref t) = self.delta {
            inputs.push(t.clone());
        }

        inputs
    }

    fn forward(&self, _scope: &mut Scope, _node_position: usize) -> TokenStream {
        let output = &self.output.name;

        // Generate values for start, limit, and delta
        let start = match &self.start {
            RangeParam::Static(value) => {
                let literal = Literal::i64_suffixed(*value);
                quote! { #literal }
            }
            RangeParam::Runtime(t) => match t {
                Type::Scalar(s) => {
                    let name = &s.name;
                    quote! { #name }
                }
                _ => panic!("Range parameter must be a scalar"),
            },
        };

        let limit = match &self.limit {
            RangeParam::Static(value) => {
                let literal = Literal::i64_suffixed(*value);
                quote! { #literal }
            }
            RangeParam::Runtime(t) => match t {
                Type::Scalar(s) => {
                    let name = &s.name;
                    quote! { #name }
                }
                _ => panic!("Range parameter must be a scalar"),
            },
        };

        let delta = match &self.delta {
            RangeParam::Static(value) => {
                let literal = Literal::i64_suffixed(*value);
                quote! { #literal }
            }
            RangeParam::Runtime(t) => match t {
                Type::Scalar(s) => {
                    let name = &s.name;
                    quote! { #name }
                }
                _ => panic!("Range parameter must be a scalar"),
            },
        };

        quote! {
            let #output = Tensor::arange_step(#start..#limit, #delta as usize, &*self.device);
        }
    }
    fn into_node(self) -> Node<PS> {
        Node::Range(self)
    }
}

impl OnnxIntoNode for RangeNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        use onnx_ir::node::range::RangeInput;
        let onnx_ir::Node::Range(n) = &node else {
            panic!("Expected Range node");
        };
        let output = TensorType::from(n.outputs.first().unwrap());

        let start = match &n.config.start {
            RangeInput::Static(value) => RangeParam::Static(*value),
            RangeInput::Runtime(runtime_ref) => {
                let arg = &n.inputs[runtime_ref.input_index];
                RangeParam::Runtime(Type::from(arg))
            }
        };

        let limit = match &n.config.limit {
            RangeInput::Static(value) => RangeParam::Static(*value),
            RangeInput::Runtime(runtime_ref) => {
                let arg = &n.inputs[runtime_ref.input_index];
                RangeParam::Runtime(Type::from(arg))
            }
        };

        let delta = match &n.config.delta {
            RangeInput::Static(value) => RangeParam::Static(*value),
            RangeInput::Runtime(runtime_ref) => {
                let arg = &n.inputs[runtime_ref.input_index];
                RangeParam::Runtime(Type::from(arg))
            }
        };

        Self::new(start, limit, delta, output)
    }
}
