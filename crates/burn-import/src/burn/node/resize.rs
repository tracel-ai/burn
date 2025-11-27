use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::node::resize::ResizeNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        use onnx_ir::node::resize::{ResizeMode, ResizeScales, ResizeSizes};

        // Only create a field for static resize (no runtime inputs)
        let has_runtime_scales = matches!(&self.config.scales, Some(ResizeScales::Runtime(_)));
        let has_runtime_sizes = matches!(&self.config.sizes, Some(ResizeSizes::Runtime(_)));

        if has_runtime_scales || has_runtime_sizes {
            return None; // Runtime resize doesn't need a field
        }

        // Check if we have static scales or sizes
        let has_static = matches!(&self.config.scales, Some(ResizeScales::Static(_)))
            || matches!(&self.config.sizes, Some(ResizeSizes::Static(_)));

        if !has_static {
            return None;
        }

        // Determine field type based on input rank
        let input_arg = self.inputs.first().unwrap();
        let input_rank = match &input_arg.ty {
            ArgType::Tensor(t) => t.rank,
            _ => panic!("Resize input must be a tensor"),
        };

        let name = syn::Ident::new(&self.name, proc_macro2::Span::call_site());
        let mode = match self.config.mode {
            ResizeMode::Nearest => quote! { burn::nn::interpolate::InterpolateMode::Nearest },
            ResizeMode::Linear => quote! { burn::nn::interpolate::InterpolateMode::Linear },
            ResizeMode::Cubic => quote! { burn::nn::interpolate::InterpolateMode::Cubic },
        };

        if input_rank == 3 {
            let size = match &self.config.sizes {
                Some(ResizeSizes::Static(sizes)) if !sizes.is_empty() => {
                    let size = sizes[0].to_tokens();
                    quote! { Some(#size) }
                }
                _ => quote! { None },
            };

            let scale_factor = match &self.config.scales {
                Some(ResizeScales::Static(scales)) if !scales.is_empty() => {
                    let scale = scales[0].to_tokens();
                    quote! { Some(#scale) }
                }
                _ => quote! { None },
            };

            Some(Field::new(
                &self.name,
                quote! { burn::nn::interpolate::Interpolate1d },
                quote! {
                    let #name = burn::nn::interpolate::Interpolate1dConfig::new()
                        .with_output_size(#size)
                        .with_scale_factor(#scale_factor)
                        .with_mode(#mode)
                        .init();
                },
            ))
        } else if input_rank == 4 {
            let size = match &self.config.sizes {
                Some(ResizeSizes::Static(sizes)) if sizes.len() >= 2 => {
                    let h = sizes[0].to_tokens();
                    let w = sizes[1].to_tokens();
                    quote! { Some([#h, #w]) }
                }
                _ => quote! { None },
            };

            let scale_factor = match &self.config.scales {
                Some(ResizeScales::Static(scales)) if scales.len() >= 2 => {
                    let h = scales[0].to_tokens();
                    let w = scales[1].to_tokens();
                    quote! { Some([#h, #w]) }
                }
                _ => quote! { None },
            };

            Some(Field::new(
                &self.name,
                quote! { burn::nn::interpolate::Interpolate2d },
                quote! {
                    let #name = burn::nn::interpolate::Interpolate2dConfig::new()
                        .with_output_size(#size)
                        .with_scale_factor(#scale_factor)
                        .with_mode(#mode)
                        .init();
                },
            ))
        } else {
            None
        }
    }

    fn field_serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        S::serialize_none(serializer)
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        use onnx_ir::node::resize::{ResizeMode, ResizeScales, ResizeSizes};

        let input_arg = self.inputs.first().unwrap();
        let input = scope.arg(input_arg);
        let output = arg_to_ident(self.outputs.first().unwrap());

        // Check if this is static (has field) or runtime resize
        let has_runtime = matches!(&self.config.scales, Some(ResizeScales::Runtime(_)))
            || matches!(&self.config.sizes, Some(ResizeSizes::Runtime(_)));
        let has_static = matches!(&self.config.scales, Some(ResizeScales::Static(_)))
            || matches!(&self.config.sizes, Some(ResizeSizes::Static(_)));

        if !has_runtime && has_static {
            // Static resize - use the field
            let field_name = syn::Ident::new(&self.name, proc_macro2::Span::call_site());
            quote! {
                let #output = self.#field_name.forward(#input);
            }
        } else {
            // Runtime resize - use tensor operations directly
            let mode = match self.config.mode {
                ResizeMode::Nearest => quote! { burn::tensor::ops::InterpolateMode::Nearest },
                ResizeMode::Linear => quote! { burn::tensor::ops::InterpolateMode::Bilinear },
                ResizeMode::Cubic => quote! { burn::tensor::ops::InterpolateMode::Bicubic },
            };

            // Handle runtime sizes input
            match &self.config.sizes {
                Some(ResizeSizes::Runtime(sizes_ref)) => {
                    let sizes_arg = &self.inputs[sizes_ref.input_index];

                    match &sizes_arg.ty {
                        ArgType::Shape(_) => {
                            let sizes_name = arg_to_ident(sizes_arg);
                            // Extract the last 2 dimensions from the shape (H, W for 2D resize)
                            quote! {
                                let #output = {
                                    let target_height = #sizes_name[2] as usize;
                                    let target_width = #sizes_name[3] as usize;
                                    burn::tensor::module::interpolate(
                                        #input,
                                        [target_height, target_width],
                                        burn::tensor::ops::InterpolateOptions::new(#mode)
                                    )
                                };
                            }
                        }
                        ArgType::Tensor(_) => {
                            let sizes_name = scope.arg(sizes_arg);
                            quote! {
                                let #output = {
                                    let sizes_data = #sizes_name.to_data().convert::<i64>();
                                    let sizes_array = sizes_data.as_slice::<i64>().unwrap();
                                    let target_height = sizes_array[2] as usize;
                                    let target_width = sizes_array[3] as usize;
                                    burn::tensor::module::interpolate(
                                        #input,
                                        [target_height, target_width],
                                        burn::tensor::ops::InterpolateOptions::new(#mode)
                                    )
                                };
                            }
                        }
                        _ => panic!("Runtime resize sizes must be Shape or Tensor"),
                    }
                }
                _ => panic!("Runtime resize requires sizes input"),
            }
        }
    }
}

#[cfg(test)]
#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use insta::assert_snapshot;
    use onnx_ir::ir::{DType, RuntimeInputRef};
    use onnx_ir::node::resize::{
        ResizeConfig, ResizeMode, ResizeNodeBuilder, ResizeScales, ResizeSizes,
    };

    // ==================== Static Resize - Rank 3 (1D) Tests ====================

    #[test]
    fn test_resize_1d_static_sizes_nearest() {
        let config = ResizeConfig {
            mode: ResizeMode::Nearest,
            scales: None,
            sizes: Some(ResizeSizes::Static(vec![64])),
            ..Default::default()
        };
        let node = ResizeNodeBuilder::new("upsample")
            .input_tensor("signal", 3, DType::F32)
            .output_tensor("upsampled", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, signal: Tensor<B, 3>) -> Tensor<B, 3> {
            let upsampled = self.upsample.forward(signal);
            upsampled
        }
        ");
    }

    #[test]
    fn test_resize_1d_static_scales_linear() {
        let config = ResizeConfig {
            mode: ResizeMode::Linear,
            scales: Some(ResizeScales::Static(vec![2.0])),
            sizes: None,
            ..Default::default()
        };
        let node = ResizeNodeBuilder::new("interpolate")
            .input_tensor("audio", 3, DType::F32)
            .output_tensor("resampled", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, audio: Tensor<B, 3>) -> Tensor<B, 3> {
            let resampled = self.interpolate.forward(audio);
            resampled
        }
        ");
    }

    #[test]
    fn test_resize_1d_static_sizes_cubic() {
        let config = ResizeConfig {
            mode: ResizeMode::Cubic,
            scales: None,
            sizes: Some(ResizeSizes::Static(vec![128])),
            ..Default::default()
        };
        let node = ResizeNodeBuilder::new("resize1d")
            .input_tensor("waveform", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, waveform: Tensor<B, 3>) -> Tensor<B, 3> {
            let output = self.resize1d.forward(waveform);
            output
        }
        ");
    }

    // ==================== Static Resize - Rank 4 (2D) Tests ====================

    #[test]
    fn test_resize_2d_static_sizes_nearest() {
        let config = ResizeConfig {
            mode: ResizeMode::Nearest,
            scales: None,
            sizes: Some(ResizeSizes::Static(vec![224, 224])),
            ..Default::default()
        };
        let node = ResizeNodeBuilder::new("resize")
            .input_tensor("image", 4, DType::F32)
            .output_tensor("resized", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, image: Tensor<B, 4>) -> Tensor<B, 4> {
            let resized = self.resize.forward(image);
            resized
        }
        ");
    }

    #[test]
    fn test_resize_2d_static_sizes_linear() {
        let config = ResizeConfig {
            mode: ResizeMode::Linear,
            scales: None,
            sizes: Some(ResizeSizes::Static(vec![512, 512])),
            ..Default::default()
        };
        let node = ResizeNodeBuilder::new("upscale")
            .input_tensor("input_img", 4, DType::F32)
            .output_tensor("output_img", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input_img: Tensor<B, 4>) -> Tensor<B, 4> {
            let output_img = self.upscale.forward(input_img);
            output_img
        }
        ");
    }

    #[test]
    fn test_resize_2d_static_sizes_cubic() {
        let config = ResizeConfig {
            mode: ResizeMode::Cubic,
            scales: None,
            sizes: Some(ResizeSizes::Static(vec![128, 256])),
            ..Default::default()
        };
        let node = ResizeNodeBuilder::new("bicubic_resize")
            .input_tensor("features", 4, DType::F32)
            .output_tensor("scaled", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, features: Tensor<B, 4>) -> Tensor<B, 4> {
            let scaled = self.bicubic_resize.forward(features);
            scaled
        }
        ");
    }

    #[test]
    fn test_resize_2d_static_scales_nearest() {
        let config = ResizeConfig {
            mode: ResizeMode::Nearest,
            scales: Some(ResizeScales::Static(vec![2.0, 2.0])),
            sizes: None,
            ..Default::default()
        };
        let node = ResizeNodeBuilder::new("double_size")
            .input_tensor("tensor", 4, DType::F32)
            .output_tensor("doubled", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, tensor: Tensor<B, 4>) -> Tensor<B, 4> {
            let doubled = self.double_size.forward(tensor);
            doubled
        }
        ");
    }

    #[test]
    fn test_resize_2d_static_scales_linear() {
        let config = ResizeConfig {
            mode: ResizeMode::Linear,
            scales: Some(ResizeScales::Static(vec![0.5, 0.5])),
            sizes: None,
            ..Default::default()
        };
        let node = ResizeNodeBuilder::new("downsample")
            .input_tensor("hires", 4, DType::F32)
            .output_tensor("lores", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, hires: Tensor<B, 4>) -> Tensor<B, 4> {
            let lores = self.downsample.forward(hires);
            lores
        }
        ");
    }

    #[test]
    fn test_resize_2d_static_scales_cubic() {
        let config = ResizeConfig {
            mode: ResizeMode::Cubic,
            scales: Some(ResizeScales::Static(vec![1.5, 1.5])),
            sizes: None,
            ..Default::default()
        };
        let node = ResizeNodeBuilder::new("scale_up")
            .input_tensor("data", 4, DType::F32)
            .output_tensor("result", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, data: Tensor<B, 4>) -> Tensor<B, 4> {
            let result = self.scale_up.forward(data);
            result
        }
        ");
    }

    // ==================== Runtime Resize with Shape Input Tests ====================

    #[test]
    fn test_resize_runtime_shape_nearest() {
        let config = ResizeConfig {
            mode: ResizeMode::Nearest,
            scales: None,
            sizes: Some(ResizeSizes::Runtime(RuntimeInputRef {
                name: "target_size".to_string(),
                input_index: 1,
            })),
            ..Default::default()
        };
        let node = ResizeNodeBuilder::new("dynamic_resize")
            .input_tensor("input", 4, DType::F32)
            .input_shape("target_size")
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 4>, target_size: [i64; 1]) -> Tensor<B, 4> {
            let output = {
                let target_height = target_size[2] as usize;
                let target_width = target_size[3] as usize;
                burn::tensor::module::interpolate(
                    input,
                    [target_height, target_width],
                    burn::tensor::ops::InterpolateOptions::new(
                        burn::tensor::ops::InterpolateMode::Nearest,
                    ),
                )
            };
            output
        }
        ");
    }

    #[test]
    fn test_resize_runtime_shape_linear() {
        let config = ResizeConfig {
            mode: ResizeMode::Linear,
            scales: None,
            sizes: Some(ResizeSizes::Runtime(RuntimeInputRef {
                name: "new_dims".to_string(),
                input_index: 1,
            })),
            ..Default::default()
        };
        let node = ResizeNodeBuilder::new("bilinear_resize")
            .input_tensor("img", 4, DType::F32)
            .input_shape("new_dims")
            .output_tensor("resized_img", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, img: Tensor<B, 4>, new_dims: [i64; 1]) -> Tensor<B, 4> {
            let resized_img = {
                let target_height = new_dims[2] as usize;
                let target_width = new_dims[3] as usize;
                burn::tensor::module::interpolate(
                    img,
                    [target_height, target_width],
                    burn::tensor::ops::InterpolateOptions::new(
                        burn::tensor::ops::InterpolateMode::Bilinear,
                    ),
                )
            };
            resized_img
        }
        ");
    }

    #[test]
    fn test_resize_runtime_shape_cubic() {
        let config = ResizeConfig {
            mode: ResizeMode::Cubic,
            scales: None,
            sizes: Some(ResizeSizes::Runtime(RuntimeInputRef {
                name: "output_shape".to_string(),
                input_index: 1,
            })),
            ..Default::default()
        };
        let node = ResizeNodeBuilder::new("cubic_interp")
            .input_tensor("source", 4, DType::F32)
            .input_shape("output_shape")
            .output_tensor("dest", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, source: Tensor<B, 4>, output_shape: [i64; 1]) -> Tensor<B, 4> {
            let dest = {
                let target_height = output_shape[2] as usize;
                let target_width = output_shape[3] as usize;
                burn::tensor::module::interpolate(
                    source,
                    [target_height, target_width],
                    burn::tensor::ops::InterpolateOptions::new(
                        burn::tensor::ops::InterpolateMode::Bicubic,
                    ),
                )
            };
            dest
        }
        ");
    }

    // ==================== Runtime Resize with Tensor Input Tests ====================

    #[test]
    fn test_resize_runtime_tensor_nearest() {
        let config = ResizeConfig {
            mode: ResizeMode::Nearest,
            scales: None,
            sizes: Some(ResizeSizes::Runtime(RuntimeInputRef {
                name: "size_tensor".to_string(),
                input_index: 1,
            })),
            ..Default::default()
        };
        let node = ResizeNodeBuilder::new("resize_op")
            .input_tensor("x", 4, DType::F32)
            .input_tensor("size_tensor", 1, DType::I64)
            .output_tensor("y", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, x: Tensor<B, 4>, size_tensor: Tensor<B, 1, Int>) -> Tensor<B, 4> {
            let y = {
                let sizes_data = size_tensor.to_data().convert::<i64>();
                let sizes_array = sizes_data.as_slice::<i64>().unwrap();
                let target_height = sizes_array[2] as usize;
                let target_width = sizes_array[3] as usize;
                burn::tensor::module::interpolate(
                    x,
                    [target_height, target_width],
                    burn::tensor::ops::InterpolateOptions::new(
                        burn::tensor::ops::InterpolateMode::Nearest,
                    ),
                )
            };
            y
        }
        ");
    }

    #[test]
    fn test_resize_runtime_tensor_linear() {
        let config = ResizeConfig {
            mode: ResizeMode::Linear,
            scales: None,
            sizes: Some(ResizeSizes::Runtime(RuntimeInputRef {
                name: "dims_tensor".to_string(),
                input_index: 1,
            })),
            ..Default::default()
        };
        let node = ResizeNodeBuilder::new("interp2d")
            .input_tensor("frame", 4, DType::F32)
            .input_tensor("dims_tensor", 1, DType::I64)
            .output_tensor("resampled_frame", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            frame: Tensor<B, 4>,
            dims_tensor: Tensor<B, 1, Int>,
        ) -> Tensor<B, 4> {
            let resampled_frame = {
                let sizes_data = dims_tensor.to_data().convert::<i64>();
                let sizes_array = sizes_data.as_slice::<i64>().unwrap();
                let target_height = sizes_array[2] as usize;
                let target_width = sizes_array[3] as usize;
                burn::tensor::module::interpolate(
                    frame,
                    [target_height, target_width],
                    burn::tensor::ops::InterpolateOptions::new(
                        burn::tensor::ops::InterpolateMode::Bilinear,
                    ),
                )
            };
            resampled_frame
        }
        ");
    }

    #[test]
    fn test_resize_runtime_tensor_cubic() {
        let config = ResizeConfig {
            mode: ResizeMode::Cubic,
            scales: None,
            sizes: Some(ResizeSizes::Runtime(RuntimeInputRef {
                name: "target_dims".to_string(),
                input_index: 1,
            })),
            ..Default::default()
        };
        let node = ResizeNodeBuilder::new("bicubic_op")
            .input_tensor("input_data", 4, DType::F32)
            .input_tensor("target_dims", 1, DType::I64)
            .output_tensor("output_data", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input_data: Tensor<B, 4>,
            target_dims: Tensor<B, 1, Int>,
        ) -> Tensor<B, 4> {
            let output_data = {
                let sizes_data = target_dims.to_data().convert::<i64>();
                let sizes_array = sizes_data.as_slice::<i64>().unwrap();
                let target_height = sizes_array[2] as usize;
                let target_width = sizes_array[3] as usize;
                burn::tensor::module::interpolate(
                    input_data,
                    [target_height, target_width],
                    burn::tensor::ops::InterpolateOptions::new(
                        burn::tensor::ops::InterpolateMode::Bicubic,
                    ),
                )
            };
            output_data
        }
        ");
    }
}
