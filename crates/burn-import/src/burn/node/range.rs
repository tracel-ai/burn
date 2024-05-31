use super::{Node, NodeCodegen};
use crate::burn::{ScalarType, Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct RangeNode {
    pub start: ScalarType,
    pub end: ScalarType,
    pub step: ScalarType,
    pub output: TensorType,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for RangeNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![
            Type::Scalar(self.start.clone()),
            Type::Scalar(self.end.clone()),
            Type::Scalar(self.step.clone()),
        ]
    }

    fn forward(&self, _scope: &mut Scope, _node_position: usize) -> TokenStream {
        let output = &self.output.name;

        let start = &self.start.name;
        let end = &self.end.name;
        let step = &self.step.name;

        quote! {
            let #output = Tensor::arange_step(#start..#end, #step as usize, &*self.device);
        }
    }
    fn into_node(self) -> Node<PS> {
        Node::Range(self)
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
    fn codegen_nodes_range() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(
            RangeNode::new(
                ScalarType::new("start", ScalarKind::Int64),
                ScalarType::new("end", ScalarKind::Int64),
                ScalarType::new("step", ScalarKind::Int64),
                TensorType::new_int("output", 1),
            )
            .into_node(),
        );
        graph.register_input_output(
            vec!["start".to_string(), "end".to_string(), "step".to_string()],
            vec!["output".to_string()],
        );

        let expected = quote! {
            use burn::tensor::Int;
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };

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
