use super::{Node, NodeCodegen};
use crate::burn::{OtherType, Scope, TensorType, ToTokens, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone)]
pub struct ResizeNode {
    pub field: OtherType,
    pub input: TensorType,
    pub output: TensorType,
    mode: String,
    scales: Vec<f32>,
    sizes: Vec<usize>,
}

impl ResizeNode {
    pub fn new<S: AsRef<str>>(
        name: S,
        input: TensorType,
        output: TensorType,
        mode: String,
        scales: Vec<f32>,
        sizes: Vec<usize>,
    ) -> Self {
        let ty = if input.dim == 3 {
            quote! {
                Interpolate1d
            }
        } else if input.dim == 4 {
            quote! {
                Interpolate2d
            }
        } else {
            panic!("Unsupported input dimension for resize node");
        };

        Self {
            field: OtherType::new(name, ty),
            input,
            output,
            mode,
            scales,
            sizes,
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for ResizeNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn field_type(&self) -> Option<Type> {
        Some(Type::Other(self.field.clone()))
    }

    fn field_init(&self) -> Option<TokenStream> {
        let name = &self.field.name;

        let mode = match self.mode.as_str() {
            "nearest" => quote! { InterpolateMode::Nearest },
            "linear" => quote! { InterpolateMode::Linear },
            "cubic" => quote! { InterpolateMode::Cubic },
            _ => panic!("Unsupported mode for resize node"),
        };

        let tokens = if self.input.dim == 3 {
            let size = if let Some(size) = self.sizes.first() {
                let size = size.to_tokens();
                quote! { Some(#size) }
            } else {
                quote! { None }
            };

            let scale_factor = if let Some(scale) = self.scales.first() {
                let scale = scale.to_tokens();
                quote! { Some(#scale) }
            } else {
                quote! { None }
            };

            quote! {
                let #name = Interpolate1dConfig::new()
                    .with_output_size(#size)
                    .with_scale_factor(#scale_factor)
                    .with_mode(#mode)
                    .init();
            }
        } else if self.input.dim == 4 {
            let size = if self.sizes.len() == 2 {
                let h = self.sizes[0].to_tokens();
                let w = self.sizes[1].to_tokens();
                quote! { Some([#h, #w]) }
            } else {
                quote! { None }
            };

            let scale_factor = if self.scales.len() == 2 {
                let h = self.scales[0].to_tokens();
                let w = self.scales[1].to_tokens();
                quote! { Some([#h, #w]) }
            } else {
                quote! { None }
            };

            quote! {
                let #name = Interpolate2dConfig::new()
                    .with_output_size(#size)
                    .with_scale_factor(#scale_factor)
                    .with_mode(#mode)
                    .init();
            }
        } else {
            panic!("Unsupported input dimension for resize node");
        };

        Some(tokens)
    }

    fn register_imports(&self, imports: &mut crate::burn::BurnImports) {
        imports.register("burn::nn::interpolate::InterpolateMode");
        if self.input.dim == 3 {
            imports.register("burn::nn::interpolate::Interpolate1dConfig");
            imports.register("burn::nn::interpolate::Interpolate1d");
        } else if self.input.dim == 4 {
            imports.register("burn::nn::interpolate::Interpolate2dConfig");
            imports.register("burn::nn::interpolate::Interpolate2d");
        } else {
            panic!("Unsupported input dimension for resize node");
        }
    }

    fn field_serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        S::serialize_none(serializer)
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;
        let field = &self.field.name;

        quote! {
            let #output = self.#field.forward(#input);
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Resize(self)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        graph::BurnGraph,
        node::{resize::ResizeNode, test::assert_tokens},
        TensorType,
    };

    #[test]
    fn test_codegen_nodes_2d() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ResizeNode::new(
            "resize",
            TensorType::new_float("tensor1", 4),
            TensorType::new_float("tensor2", 4),
            "nearest".to_string(),
            vec![0.5, 0.5],
            vec![],
        ));

        graph.register_input_output(vec!["tensor1".to_string()], vec!["tensor2".to_string()]);

        let expected = quote! {
            use burn::nn::interpolate::Interpolate2d;
            use burn::nn::interpolate::Interpolate2dConfig;
            use burn::nn::interpolate::InterpolateMode;
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };
            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                resize: Interpolate2d,
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }
            impl<B: Backend> Model<B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    let resize = Interpolate2dConfig::new()
                        .with_output_size(None)
                        .with_scale_factor(Some([0.5, 0.5]))
                        .with_mode(InterpolateMode::Nearest)
                        .init();
                    Self {
                        resize,
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = self.resize.forward(tensor1);
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_nodes_1d() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ResizeNode::new(
            "resize",
            TensorType::new_float("tensor1", 3),
            TensorType::new_float("tensor2", 3),
            "cubic".to_string(),
            vec![2.0],
            vec![20],
        ));

        graph.register_input_output(vec!["tensor1".to_string()], vec!["tensor2".to_string()]);

        let expected = quote! {
            use burn::nn::interpolate::Interpolate1d;
            use burn::nn::interpolate::Interpolate1dConfig;
            use burn::nn::interpolate::InterpolateMode;
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };
            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                resize: Interpolate1d,
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }
            impl<B: Backend> Model<B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    let resize = Interpolate1dConfig::new()
                        .with_output_size(Some(20))
                        .with_scale_factor(Some(2.0))
                        .with_mode(InterpolateMode::Cubic)
                        .init();
                    Self {
                        resize,
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, tensor1: Tensor<B, 3>) -> Tensor<B, 3> {
                    let tensor2 = self.resize.forward(tensor1);
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
