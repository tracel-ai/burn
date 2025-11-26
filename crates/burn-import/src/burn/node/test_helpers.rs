//! Test helpers for node code generation tests
//!
//! This module provides common utilities for testing node code generation,
//! making tests more concise and maintainable.

use super::NodeCodegen;
use crate::burn::Scope;
use crate::burn::argument_helpers::arg_type_tokens;
use burn::record::PrecisionSettings;
use onnx_ir::ir::ArgType;
use proc_macro2::{Ident, Span};
use quote::quote;

/// Generate forward pass code for a node with optional clone behavior
///
/// Generates a complete forward function signature with inputs, outputs, and body.
///
/// # Arguments
/// * `node` - The node to generate code for
/// * `input_idx` - Index of the input to register (default: 0)
/// * `with_clone` - Whether to register future use to trigger clone
/// * `node_position` - Position of the node in the graph (default: 1)
///
/// # Example
/// ```ignore
/// let node = create_my_node("test");
/// let code = codegen_forward(&node, 0, false, 1);
/// assert_snapshot!(code, @r#"
///     pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
///         let output = input.abs();
///         output
///     }
/// "#);
/// ```
pub fn codegen_forward<T, PS>(
    node: &T,
    _input_idx: usize,
    with_clone: bool,
    node_position: usize,
) -> String
where
    T: NodeCodegen<PS>,
    PS: PrecisionSettings,
{
    let mut scope = Scope::default();

    // Register all inputs as variables
    for input in node.inputs().iter() {
        // Skip non-dynamic inputs (constants, initializers)
        if !matches!(
            input.ty,
            ArgType::Tensor(_) | ArgType::Scalar(_) | ArgType::Shape(_)
        ) {
            continue;
        }
        scope.tensor_register_variable(input, 0);

        if with_clone {
            // Register two future uses to trigger clone
            scope.tensor_register_future_use(input, node_position);
            scope.tensor_register_future_use(input, node_position + 1);
        }
    }

    // Generate code using the node's forward method with ScopeAtPosition
    let mut scope_at_pos = scope.at_position(node_position);
    let body = node.forward(&mut scope_at_pos);

    // Generate function parameters from inputs
    let params: Vec<_> = node
        .inputs()
        .iter()
        .filter(|arg| {
            // Only include dynamic inputs (not constants/initializers)
            matches!(
                arg.ty,
                ArgType::Tensor(_) | ArgType::Scalar(_) | ArgType::Shape(_)
            ) && arg.value().is_none()
        })
        .map(|arg| {
            let name = Ident::new(&arg.name, Span::call_site());
            let ty = arg_type_tokens(arg);
            quote! { #name: #ty }
        })
        .collect();

    // Generate return type from outputs
    let outputs = node.outputs();
    let return_type = if outputs.len() == 1 {
        arg_type_tokens(&outputs[0])
    } else {
        let types: Vec<_> = outputs.iter().map(arg_type_tokens).collect();
        quote! { (#(#types),*) }
    };

    // Generate return expression
    let return_expr = if outputs.len() == 1 {
        let name = Ident::new(&outputs[0].name, Span::call_site());
        quote! { #name }
    } else {
        let names: Vec<_> = outputs
            .iter()
            .map(|arg| Ident::new(&arg.name, Span::call_site()))
            .collect();
        quote! { (#(#names),*) }
    };

    // Generate the full forward function
    let forward_fn = quote! {
        pub fn forward(&self, #(#params),*) -> #return_type {
            #body
            #return_expr
        }
    };

    format_tokens(forward_fn)
}

/// Generate forward pass code with default parameters
///
/// Uses FullPrecisionSettings and:
/// - input_idx: 0 (first input)
/// - with_clone: false (no clone)
/// - node_position: 1
///
/// # Example
/// ```ignore
/// let node = create_my_node("test");
/// let code = codegen_forward_default(&node);
/// assert_snapshot!(code, @r#"
///     pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
///         let output = input.abs();
///         output
///     }
/// "#);
/// ```
pub fn codegen_forward_default<T>(node: &T) -> String
where
    T: NodeCodegen<burn::record::FullPrecisionSettings>,
{
    codegen_forward(node, 0, false, 1)
}

/// Generate forward pass code with clone enabled
///
/// Uses FullPrecisionSettings and:
/// - input_idx: 0 (first input)
/// - with_clone: true (triggers clone)
/// - node_position: 1
///
/// # Example
/// ```ignore
/// let node = create_my_node("test");
/// let code = codegen_forward_with_clone(&node);
/// assert!(code.contains("clone"));
/// ```
pub fn codegen_forward_with_clone<T>(node: &T) -> String
where
    T: NodeCodegen<burn::record::FullPrecisionSettings>,
{
    codegen_forward(node, 0, true, 1)
}

/// Format a TokenStream using PrettyPlease
fn format_tokens(tokens: proc_macro2::TokenStream) -> String {
    use rust_format::{Config, Formatter, PostProcess, PrettyPlease};

    let config = Config::new_str().post_proc(PostProcess::ReplaceMarkersAndDocBlocks);
    let formatter = PrettyPlease::from_config(config);
    let fallback = tokens.to_string();
    formatter.format_tokens(tokens).unwrap_or(fallback)
}
