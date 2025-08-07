use super::{Node, NodeCodegen};
use crate::burn::{BurnImports, Scope, TensorType, ToTokens, Type};
use burn::record::PrecisionSettings;
use onnx_ir::node::unsqueeze::UnsqueezeConfig;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct UnsqueezeNode {
    pub input: Type,
    pub output: TensorType,
    pub axes: UnsqueezeConfig,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for UnsqueezeNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<Type> {
        let input = self.input.clone();
        match &self.axes {
            UnsqueezeConfig::Static(_) => vec![input],
            UnsqueezeConfig::Runtime(rt_type) => vec![input, Type::from(rt_type)],
        }
    }
    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let output_name = &self.output.name;
        let output_rank = self.output.rank.to_tokens();

        let axes = match &self.axes {
            UnsqueezeConfig::Static(static_axes) => static_axes.to_tokens(),
            UnsqueezeConfig::Runtime(arg) => match Type::from(arg) {
                Type::Tensor(axes_tensor) => {
                    let tensor_name = &axes_tensor.name;
                    quote! {
                        #tensor_name.to_data().as_slice::<B::IntElem>().unwrap().iter().map(|&x| x.to_isize()).collect::<Vec<isize>>()
                    }
                }
                _ => panic!(
                    "UnsqueezeNode received invalid axes type: expected tensor but got {arg:?}"
                ),
            },
        };

        match &self.input {
            Type::Tensor(tensor) => {
                let input = scope.tensor_use_owned(tensor, node_position);
                quote! {
                    let #output_name: Tensor<B, #output_rank> = #input.unsqueeze_dims(&#axes);
                }
            }
            Type::Scalar(scalar) => {
                let scalar_name = &scalar.name;
                quote! {
                    let #output_name = Tensor::<B, #output_rank>::from_data([#scalar_name.elem::<B::FloatElem>()], &self.device).unsqueeze();
                }
            }
            _ => panic!(
                "UnsqueezeNode received unsupported input type: expected tensor or scalar but got {:?}",
                self.input
            ),
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Unsqueeze(self)
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        match &self.input {
            Type::Scalar(_) => {
                imports.register("burn::tensor::ElementConversion");
            }
            _ => {}
        }
        match &self.axes {
            UnsqueezeConfig::Runtime(_) => {
                imports.register("alloc::vec::Vec");
                imports.register("burn::tensor::cast::ToElement");
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        TensorType, Type,
        graph::BurnGraph,
        node::{test::assert_tokens, unsqueeze::UnsqueezeNode},
    };

    #[test]
    fn test_codegen_nodes() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(UnsqueezeNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 3)),
            TensorType::new_float("tensor2", 5),
            UnsqueezeConfig::Static([0, 4].into()),
        ));

        graph.register_input_output(vec!["tensor1".to_string()], vec!["tensor2".to_string()]);

        let expected = quote! {
            use burn::tensor::Tensor;
            use burn::{
                module::Module,
                tensor::backend::Backend,
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
                pub fn forward(&self, tensor1: Tensor<B, 3>) -> Tensor<B, 5> {
                    let tensor2: Tensor<B, 5> = tensor1.unsqueeze_dims(&[0,4]);
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
