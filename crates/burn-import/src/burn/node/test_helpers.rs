//! Test helpers for node code generation tests
//!
//! This module provides common utilities for testing node code generation,
//! making tests more concise and maintainable.

use super::NodeCodegen;
use crate::burn::Scope;
use crate::burn::argument_helpers::{codegen_fn_params, codegen_return_expr, codegen_return_type};
use onnx_ir::ir::ArgType;
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
pub fn codegen_forward<T>(
    node: &T,
    _input_idx: usize,
    with_clone: bool,
    node_position: usize,
) -> String
where
    T: NodeCodegen,
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

    // Filter inputs to only include dynamic inputs (not constants/initializers)
    let dynamic_inputs: Vec<_> = node
        .inputs()
        .iter()
        .filter(|arg| {
            matches!(
                arg.ty,
                ArgType::Tensor(_) | ArgType::Scalar(_) | ArgType::Shape(_)
            ) && arg.value().is_none()
        })
        .cloned()
        .collect();

    // Use shared helpers for generating function signature parts
    let input_def = codegen_fn_params(&dynamic_inputs);
    let return_type = codegen_return_type(node.outputs());
    let return_expr = codegen_return_expr(node.outputs());

    // Generate the full forward function
    let forward_fn = quote! {
        pub fn forward(&self, #input_def) -> #return_type {
            #body
            #return_expr
        }
    };

    format_tokens(forward_fn)
}

/// Generate forward pass code with default parameters
///
/// Uses:
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
    T: NodeCodegen,
{
    codegen_forward(node, 0, false, 1)
}

/// Generate forward pass code with clone enabled
///
/// Uses:
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
    T: NodeCodegen,
{
    codegen_forward(node, 0, true, 1)
}

/// Generate field initialization code for a node
///
/// Returns the initialization code from the node's field() method.
///
/// # Example
/// ```ignore
/// let node = create_pool_node("pool1", true);
/// let code = codegen_field_init(&node);
/// assert!(code.contains(".with_ceil_mode(true)"));
/// ```
pub fn codegen_field_init<T>(node: &T) -> String
where
    T: NodeCodegen,
{
    if let Some(field) = node.field() {
        format_statement(field.init)
    } else {
        String::new()
    }
}

/// Format a TokenStream using PrettyPlease
fn format_tokens(tokens: proc_macro2::TokenStream) -> String {
    use rust_format::{Config, Formatter, PostProcess, PrettyPlease};

    let config = Config::new_str().post_proc(PostProcess::ReplaceMarkersAndDocBlocks);
    let formatter = PrettyPlease::from_config(config);
    let fallback = tokens.to_string();
    formatter.format_tokens(tokens).unwrap_or(fallback)
}

/// Format a statement TokenStream by wrapping it in a function
fn format_statement(tokens: proc_macro2::TokenStream) -> String {
    use rust_format::{Config, Formatter, PostProcess, PrettyPlease};

    // Wrap in a function to make it valid syntax for PrettyPlease
    let wrapped = quote! {
        fn __wrapper() {
            #tokens
        }
    };

    let config = Config::new_str().post_proc(PostProcess::ReplaceMarkersAndDocBlocks);
    let formatter = PrettyPlease::from_config(config);

    match formatter.format_tokens(wrapped) {
        Ok(formatted) => {
            // Extract the body (remove the wrapper function)
            let lines: Vec<&str> = formatted.lines().collect();
            // Skip "fn __wrapper() {" and the closing "}"
            let body_lines: Vec<&str> = lines
                .iter()
                .skip(1)
                .take(lines.len().saturating_sub(2))
                .copied()
                .collect();
            // Remove one level of indentation (4 spaces)
            body_lines
                .iter()
                .map(|line| {
                    if line.starts_with("    ") {
                        &line[4..]
                    } else {
                        *line
                    }
                })
                .collect::<Vec<_>>()
                .join("\n")
        }
        Err(_) => tokens.to_string(),
    }
}
