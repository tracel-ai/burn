use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{OtherType, Scope, TensorType, ToTokens, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

/// Interpolation mode for resize operation
#[derive(Debug, Clone, PartialEq)]
pub enum ResizeMode {
    /// Nearest neighbor interpolation
    Nearest,
    /// Linear interpolation (bilinear for 2D, trilinear for 3D)
    Linear,
    /// Cubic interpolation
    Cubic,
}

impl ResizeMode {
    /// Convert to InterpolateMode token for code generation
    pub fn to_interpolate_mode_token(&self) -> TokenStream {
        match self {
            ResizeMode::Nearest => quote! { InterpolateMode::Nearest },
            ResizeMode::Linear => quote! { InterpolateMode::Linear },
            ResizeMode::Cubic => quote! { InterpolateMode::Cubic },
        }
    }

    /// Convert to tensor ops InterpolateMode token for runtime resize
    pub fn to_tensor_interpolate_mode_token(&self) -> TokenStream {
        match self {
            ResizeMode::Nearest => quote! { burn::tensor::ops::InterpolateMode::Nearest },
            ResizeMode::Linear => quote! { burn::tensor::ops::InterpolateMode::Bilinear },
            ResizeMode::Cubic => quote! { burn::tensor::ops::InterpolateMode::Bicubic },
        }
    }
}

#[derive(Debug, Clone)]
pub enum ResizeScales {
    Static(Vec<f32>),
    Runtime(Type),
}

#[derive(Debug, Clone)]
pub enum ResizeSizes {
    Static(Vec<usize>),
    Runtime(Type),
}

#[derive(Debug, Clone)]
pub struct ResizeNode {
    pub field: Option<OtherType>,
    pub input: TensorType,
    pub output: TensorType,
    pub mode: ResizeMode,
    pub scales: Option<ResizeScales>,
    pub sizes: Option<ResizeSizes>,
}

impl ResizeNode {
    pub fn new<S: AsRef<str>>(
        name: S,
        input: TensorType,
        output: TensorType,
        mode: ResizeMode,
        scales: Option<ResizeScales>,
        sizes: Option<ResizeSizes>,
    ) -> Self {
        // Only create a field if we have static scales/sizes
        // Runtime inputs mean we don't need a static interpolation field
        let field = match (&scales, &sizes) {
            (Some(ResizeScales::Runtime(_)), _) | (_, Some(ResizeSizes::Runtime(_))) => {
                None // Runtime inputs, no field needed
            }
            (Some(ResizeScales::Static(_)), _) | (_, Some(ResizeSizes::Static(_))) => {
                let ty = if input.rank == 3 {
                    quote! { Interpolate1d }
                } else if input.rank == 4 {
                    quote! { Interpolate2d }
                } else {
                    panic!("Unsupported input rank for resize node");
                };
                Some(OtherType::new(name, ty))
            }
            _ => None, // No scales or sizes provided
        };

        Self {
            field,
            input,
            output,
            mode,
            scales,
            sizes,
        }
    }

    fn forward_runtime(&self, input: TokenStream, output: &proc_macro2::Ident) -> TokenStream {
        // Handle runtime resize with Shape inputs
        match &self.sizes {
            Some(ResizeSizes::Runtime(Type::Shape(shape))) => {
                let shape_name = &shape.name;
                // Extract the last 2 dimensions from the shape (H, W for 2D resize)
                if self.input.rank == 4 {
                    let mode_token = self.mode.to_tensor_interpolate_mode_token();

                    quote! {
                        // Shape contains the full output shape [N, C, H, W]
                        // We need to extract H and W for the resize operation
                        let target_height = #shape_name[2] as usize;
                        let target_width = #shape_name[3] as usize;

                        // Use interpolate function directly
                        let #output = burn::tensor::module::interpolate(
                            #input,
                            [target_height, target_width],
                            burn::tensor::ops::InterpolateOptions::new(#mode_token)
                        );
                    }
                } else {
                    panic!(
                        "Runtime resize with Shape input only supported for 4D tensors currently"
                    );
                }
            }
            Some(ResizeSizes::Runtime(Type::Tensor(tensor))) => {
                let sizes_name = &tensor.name;
                let mode_token = self.mode.to_tensor_interpolate_mode_token();
                quote! {
                    // Convert tensor to shape array, forcing conversion to i64
                    let sizes_data = #sizes_name.to_data().convert::<i64>();
                    let sizes_array = sizes_data.as_slice::<i64>().unwrap();
                    let target_height = sizes_array[2] as usize;
                    let target_width = sizes_array[3] as usize;

                    let #output = burn::tensor::module::interpolate(
                        #input,
                        [target_height, target_width],
                        burn::tensor::ops::InterpolateOptions::new(#mode_token)
                    );
                }
            }
            _ => panic!("Runtime resize requires Shape or Tensor sizes input"),
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for ResizeNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<Type> {
        let mut types = vec![Type::Tensor(self.input.clone())];

        // Add runtime shape inputs if present
        if let Some(ResizeScales::Runtime(ty)) = &self.scales {
            types.push(ty.clone());
        }
        if let Some(ResizeSizes::Runtime(ty)) = &self.sizes {
            types.push(ty.clone());
        }

        types
    }

    fn field_type(&self) -> Option<Type> {
        self.field.as_ref().map(|f| Type::Other(f.clone()))
    }

    fn field_init(&self) -> Option<TokenStream> {
        let field = self.field.as_ref()?;
        let name = &field.name;

        let mode = self.mode.to_interpolate_mode_token();

        let tokens = if self.input.rank == 3 {
            let size = match &self.sizes {
                Some(ResizeSizes::Static(sizes)) if !sizes.is_empty() => {
                    let size = sizes[0].to_tokens();
                    quote! { Some(#size) }
                }
                _ => quote! { None },
            };

            let scale_factor = match &self.scales {
                Some(ResizeScales::Static(scales)) if !scales.is_empty() => {
                    let scale = scales[0].to_tokens();
                    quote! { Some(#scale) }
                }
                _ => quote! { None },
            };

            quote! {
                let #name = Interpolate1dConfig::new()
                    .with_output_size(#size)
                    .with_scale_factor(#scale_factor)
                    .with_mode(#mode)
                    .init();
            }
        } else if self.input.rank == 4 {
            let size = match &self.sizes {
                Some(ResizeSizes::Static(sizes)) if sizes.len() == 2 => {
                    let h = sizes[0].to_tokens();
                    let w = sizes[1].to_tokens();
                    quote! { Some([#h, #w]) }
                }
                _ => quote! { None },
            };

            let scale_factor = match &self.scales {
                Some(ResizeScales::Static(scales)) if scales.len() == 2 => {
                    let h = scales[0].to_tokens();
                    let w = scales[1].to_tokens();
                    quote! { Some([#h, #w]) }
                }
                _ => quote! { None },
            };

            quote! {
                let #name = Interpolate2dConfig::new()
                    .with_output_size(#size)
                    .with_scale_factor(#scale_factor)
                    .with_mode(#mode)
                    .init();
            }
        } else {
            panic!("Unsupported input rank for resize node");
        };

        Some(tokens)
    }

    fn register_imports(&self, imports: &mut crate::burn::BurnImports) {
        if self.field.is_some() {
            // Static resize - need the interpolate config and types
            imports.register("burn::nn::interpolate::InterpolateMode");
            if self.input.rank == 3 {
                imports.register("burn::nn::interpolate::Interpolate1dConfig");
                imports.register("burn::nn::interpolate::Interpolate1d");
            } else if self.input.rank == 4 {
                imports.register("burn::nn::interpolate::Interpolate2dConfig");
                imports.register("burn::nn::interpolate::Interpolate2d");
            } else {
                panic!("Unsupported input rank for resize node");
            }
        }
    }

    fn field_serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        S::serialize_none(serializer)
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;

        if let Some(field) = &self.field {
            let field_name = &field.name;
            quote! {
                let #output = self.#field_name.forward(#input);
            }
        } else {
            // Handle runtime resize - we need to use tensor operations directly
            self.forward_runtime(input, output)
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Resize(self)
    }
}

impl OnnxIntoNode for ResizeNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let (inputs, outputs, config, name) = match &node {
            onnx_ir::Node::Resize {
                inputs,
                outputs,
                config,
                name,
                ..
            } => (inputs, outputs, config, name),
            _ => panic!("Expected Resize node"),
        };

        let input = TensorType::from(&inputs[0]);
        let output = TensorType::from(outputs.first().unwrap());

        // Convert from onnx-ir types to burn types
        let mode = match config.mode {
            onnx_ir::node::resize::ResizeMode::Nearest => ResizeMode::Nearest,
            onnx_ir::node::resize::ResizeMode::Linear => ResizeMode::Linear,
            onnx_ir::node::resize::ResizeMode::Cubic => ResizeMode::Cubic,
        };

        let scales = match &config.scales {
            Some(onnx_ir::node::resize::ResizeScales::Static(s)) => {
                Some(ResizeScales::Static(s.clone()))
            }
            Some(onnx_ir::node::resize::ResizeScales::Runtime(scales_ref)) => {
                // Get the actual argument using the RuntimeInputRef
                let scales_arg = &inputs[scales_ref.input_index];
                Some(ResizeScales::Runtime(Type::from(scales_arg)))
            }
            None => None,
        };

        let sizes = match &config.sizes {
            Some(onnx_ir::node::resize::ResizeSizes::Static(s)) => {
                Some(ResizeSizes::Static(s.clone()))
            }
            Some(onnx_ir::node::resize::ResizeSizes::Runtime(sizes_ref)) => {
                // Get the actual argument using the RuntimeInputRef
                let sizes_arg = &inputs[sizes_ref.input_index];
                Some(ResizeSizes::Runtime(Type::from(sizes_arg)))
            }
            None => None,
        };

        ResizeNode::new(name, input, output, mode, scales, sizes)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        TensorType,
        graph::BurnGraph,
        node::{resize::ResizeNode, test::assert_tokens},
    };

    #[test]
    fn test_codegen_nodes_2d() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ResizeNode::new(
            "resize",
            TensorType::new_float("tensor1", 4),
            TensorType::new_float("tensor2", 4),
            ResizeMode::Nearest,
            Some(ResizeScales::Static(vec![0.5, 0.5])),
            None,
        ));

        graph.register_input_output(
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
            &[],
            &[],
        );

        let expected = quote! {
            use burn::prelude::*;
            use burn::nn::interpolate::Interpolate2d;
            use burn::nn::interpolate::Interpolate2dConfig;
            use burn::nn::interpolate::InterpolateMode;
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
            ResizeMode::Cubic,
            Some(ResizeScales::Static(vec![2.0])),
            Some(ResizeSizes::Static(vec![20])),
        ));

        graph.register_input_output(
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
            &[],
            &[],
        );

        let expected = quote! {
            use burn::prelude::*;
            use burn::nn::interpolate::Interpolate1d;
            use burn::nn::interpolate::Interpolate1dConfig;
            use burn::nn::interpolate::InterpolateMode;
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
