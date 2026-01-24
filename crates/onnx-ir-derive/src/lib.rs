#![cfg_attr(docsrs, feature(doc_cfg))]

use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, parse_macro_input};

/// Derive macro for generating node builders
///
/// Automatically generates a builder with methods for constructing node inputs/outputs.
///
/// # Example
/// ```ignore
/// #[derive(Debug, Clone, NodeBuilder)]
/// pub struct AddNode {
///     pub name: String,
///     pub inputs: Vec<Argument>,
///     pub outputs: Vec<Argument>,
/// }
/// ```
///
/// Generates `AddNodeBuilder` with:
/// - `new(name)` - Create builder
/// - `input_tensor(name, rank, dtype)` - Add tensor input (dynamic, no static shape)
/// - `input_tensor_shape(name, shape, dtype)` - Add tensor input with static shape
/// - `input_scalar(name, dtype)` - Add scalar input
/// - `input_shape(name)` - Add shape input
/// - `output_tensor(name, rank, dtype)` - Add output tensor
/// - `output_scalar(name, dtype)` - Add scalar output
/// - `output_shape(name)` - Add shape output
/// - `config(config)` - Set config (if node has a config field)
/// - `build()` - Build the node
#[proc_macro_derive(NodeBuilder)]
pub fn node_builder_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let node_name = &input.ident;
    let builder_name = syn::Ident::new(&format!("{}Builder", node_name), node_name.span());

    // Check if the struct has a config field
    let has_config = if let Data::Struct(data) = &input.data {
        if let Fields::Named(fields) = &data.fields {
            fields
                .named
                .iter()
                .any(|f| f.ident.as_ref().map(|i| i == "config").unwrap_or(false))
        } else {
            false
        }
    } else {
        false
    };

    // Extract config type if it exists
    let config_type = if has_config {
        if let Data::Struct(data) = &input.data {
            if let Fields::Named(fields) = &data.fields {
                fields
                    .named
                    .iter()
                    .find(|f| f.ident.as_ref().map(|i| i == "config").unwrap_or(false))
                    .map(|f| &f.ty)
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    let config_field = if let Some(config_ty) = config_type {
        quote! {
            config: Option<#config_ty>,
        }
    } else {
        quote! {}
    };

    let config_init = if has_config {
        quote! { config: None, }
    } else {
        quote! {}
    };

    let config_method = if let Some(config_ty) = config_type {
        quote! {
            /// Set the configuration
            pub fn config(mut self, config: #config_ty) -> Self {
                self.config = Some(config);
                self
            }
        }
    } else {
        quote! {}
    };

    let config_build = if has_config {
        quote! {
            config: self.config.expect("Config must be set before calling build()"),
        }
    } else {
        quote! {}
    };

    let expanded = quote! {
        pub struct #builder_name {
            name: String,
            inputs: Vec<crate::ir::Argument>,
            outputs: Vec<crate::ir::Argument>,
            #config_field
        }

        impl #builder_name {
            /// Create a new builder
            pub fn new(name: impl Into<String>) -> Self {
                Self {
                    name: name.into(),
                    inputs: vec![],
                    outputs: vec![],
                    #config_init
                }
            }

            /// Add a tensor input (dynamic, no static shape)
            pub fn input_tensor(
                mut self,
                name: &str,
                rank: usize,
                dtype: burn_tensor::DType,
            ) -> Self {
                use crate::ir::{Argument, ArgType, TensorType};
                self.inputs.push(Argument::new(
                    name,
                    ArgType::Tensor(TensorType {
                        dtype,
                        rank,
                        static_shape: None,
                    }),
                ));
                self
            }

            /// Add a tensor input with static shape
            pub fn input_tensor_shape(
                mut self,
                name: &str,
                shape: Vec<usize>,
                dtype: burn_tensor::DType,
            ) -> Self {
                use crate::ir::{Argument, ArgType, TensorType};
                self.inputs.push(Argument::new(
                    name,
                    ArgType::Tensor(TensorType {
                        dtype,
                        rank: shape.len(),
                        static_shape: Some(shape),
                    }),
                ));
                self
            }

            /// Add a scalar input
            pub fn input_scalar(mut self, name: &str, dtype: burn_tensor::DType) -> Self {
                use crate::ir::{Argument, ArgType};
                self.inputs.push(Argument::new(name, ArgType::Scalar(dtype)));
                self
            }

            /// Add a shape input (rank 1 by default, since shapes are 1D arrays)
            pub fn input_shape(mut self, name: &str) -> Self {
                use crate::ir::{Argument, ArgType};
                self.inputs.push(Argument::new(name, ArgType::Shape(1)));
                self
            }

            /// Add an output tensor
            pub fn output_tensor(
                mut self,
                name: &str,
                rank: usize,
                dtype: burn_tensor::DType,
            ) -> Self {
                use crate::ir::{Argument, ArgType, TensorType};
                self.outputs.push(Argument::new(
                    name,
                    ArgType::Tensor(TensorType {
                        dtype,
                        rank,
                        static_shape: None,
                    }),
                ));
                self
            }

            /// Add a scalar output
            pub fn output_scalar(mut self, name: &str, dtype: burn_tensor::DType) -> Self {
                use crate::ir::{Argument, ArgType};
                self.outputs.push(Argument::new(name, ArgType::Scalar(dtype)));
                self
            }

            /// Add a shape output (size 1 by default, since shapes are 1D arrays of length 1)
            pub fn output_shape(mut self, name: &str) -> Self {
                use crate::ir::{Argument, ArgType};
                self.outputs.push(Argument::new(name, ArgType::Shape(1)));
                self
            }

            /// Add a shape output with a specific size (number of elements in the shape array)
            pub fn output_shape_with_size(mut self, name: &str, size: usize) -> Self {
                use crate::ir::{Argument, ArgType};
                self.outputs.push(Argument::new(name, ArgType::Shape(size)));
                self
            }

            /// Add a constant i64 scalar input with a known value
            pub fn input_const_i64(mut self, name: &str, value: i64) -> Self {
                use crate::ir::Argument;
                let arg = Argument::from_const_i64(name, value);
                self.inputs.push(arg);
                self
            }

            #config_method

            /// Build the node
            pub fn build(self) -> #node_name {
                #node_name {
                    name: self.name,
                    inputs: self.inputs,
                    outputs: self.outputs,
                    #config_build
                }
            }
        }
    };

    TokenStream::from(expanded)
}
