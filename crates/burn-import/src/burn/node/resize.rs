use super::{NodeCodegen, arg_to_ident};
use crate::burn::{Field, Scope, ToTokens};
use burn::record::PrecisionSettings;
use onnx_ir::{ArgType, Argument};
use proc_macro2::TokenStream;
use quote::quote;

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

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        use onnx_ir::node::resize::{ResizeMode, ResizeScales, ResizeSizes};

        let input_arg = self.inputs.first().unwrap();
        let input = scope.tensor_use_owned(input_arg, node_position);
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
                                let target_height = #sizes_name[2] as usize;
                                let target_width = #sizes_name[3] as usize;

                                let #output = burn::tensor::module::interpolate(
                                    #input,
                                    [target_height, target_width],
                                    burn::tensor::ops::InterpolateOptions::new(#mode)
                                );
                            }
                        }
                        ArgType::Tensor(_) => {
                            let sizes_name = scope.tensor_use_owned(sizes_arg, node_position);
                            quote! {
                                let sizes_data = #sizes_name.to_data().convert::<i64>();
                                let sizes_array = sizes_data.as_slice::<i64>().unwrap();
                                let target_height = sizes_array[2] as usize;
                                let target_width = sizes_array[3] as usize;

                                let #output = burn::tensor::module::interpolate(
                                    #input,
                                    [target_height, target_width],
                                    burn::tensor::ops::InterpolateOptions::new(#mode)
                                );
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
