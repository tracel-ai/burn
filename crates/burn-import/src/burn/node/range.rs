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
        let config = onnx_ir::node::range::range_config(&node);
        let output = TensorType::from(node.outputs.first().unwrap());

        let start = match config.start {
            RangeInput::Static(value) => RangeParam::Static(value),
            RangeInput::Runtime(arg) => RangeParam::Runtime(Type::from(&arg)),
        };

        let limit = match config.limit {
            RangeInput::Static(value) => RangeParam::Static(value),
            RangeInput::Runtime(arg) => RangeParam::Runtime(Type::from(&arg)),
        };

        let delta = match config.delta {
            RangeInput::Static(value) => RangeParam::Static(value),
            RangeInput::Runtime(arg) => RangeParam::Runtime(Type::from(&arg)),
        };

        Self::new(start, limit, delta, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::graph::BurnGraph;
    use crate::burn::node::test::assert_tokens;
    use crate::burn::{ScalarKind, ScalarType};
    use burn::record::FullPrecisionSettings;

    #[test]
    fn codegen_nodes_range_static() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(
            RangeNode::new(
                RangeParam::Static(0),
                RangeParam::Static(10),
                RangeParam::Static(2),
                TensorType::new_int("output", 1),
            )
            .into_node(),
        );
        graph.register_input_output(vec![], vec!["output".to_string()]);

        let expected = quote! {
            use burn::prelude::*;

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self) -> Tensor<B, 1, Int> {
                    let output = Tensor::arange_step(0i64..10i64, 2i64 as usize, &*self.device);

                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn codegen_nodes_range_runtime() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(
            RangeNode::new(
                RangeParam::Runtime(Type::Scalar(ScalarType::new("start", ScalarKind::Int64))),
                RangeParam::Runtime(Type::Scalar(ScalarType::new("end", ScalarKind::Int64))),
                RangeParam::Runtime(Type::Scalar(ScalarType::new("step", ScalarKind::Int64))),
                TensorType::new_int("output", 1),
            )
            .into_node(),
        );
        graph.register_input_output(
            vec!["start".to_string(), "end".to_string(), "step".to_string()],
            vec!["output".to_string()],
        );

        let expected = quote! {
            use burn::prelude::*;

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, start: i64, end: i64, step: i64) -> Tensor<B, 1, Int> {
                    let output = Tensor::arange_step(start..end, step as usize, &*self.device);

                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
