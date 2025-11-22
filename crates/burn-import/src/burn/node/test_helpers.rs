//! Test helpers for node code generation tests
//!
//! This module provides common utilities for testing node code generation,
//! making tests more concise and maintainable.

use super::NodeCodegen;
use crate::burn::Scope;
use burn::record::PrecisionSettings;

/// Generate forward pass code for a node with optional clone behavior
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
/// assert_snapshot!(code, @"...");
/// ```
pub fn codegen_forward<T, PS>(
    node: &T,
    input_idx: usize,
    with_clone: bool,
    node_position: usize,
) -> String
where
    T: NodeCodegen<PS>,
    PS: PrecisionSettings,
{
    let mut scope = Scope::default();

    if let Some(input) = node.inputs().get(input_idx) {
        // Register the variable at position 0
        scope.tensor_register_variable(input, 0);

        if with_clone {
            // Register two future uses to trigger clone:
            // 1. The current use (at node_position)
            // 2. A future use (after node_position) to ensure clone is needed
            scope.tensor_register_future_use(input, node_position);
            scope.tensor_register_future_use(input, node_position + 1);
        }
    }

    // Generate code using the node's forward method with ScopeAtPosition
    let mut scope_at_pos = scope.at_position(node_position);
    let code = node.forward(&mut scope_at_pos);

    // Format the statement by wrapping in a function, formatting, then extracting
    format_statement(code)
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
/// assert_snapshot!(code, @"...");
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

/// Format a statement-level TokenStream by wrapping it in a function,
/// formatting, then extracting the statement
fn format_statement(stmt: proc_macro2::TokenStream) -> String {
    use quote::quote;
    use rust_format::{Config, Formatter, PostProcess, PrettyPlease};

    // Wrap statement in a dummy function
    let wrapped = quote! {
        fn __fmt() {
            #stmt
        }
    };

    // Format the function
    let config = Config::new_str().post_proc(PostProcess::ReplaceMarkersAndDocBlocks);
    let formatter = PrettyPlease::from_config(config);
    let formatted = formatter
        .format_tokens(wrapped)
        .unwrap_or_else(|_| stmt.to_string());

    // Extract just the statement (remove function wrapper)
    // The formatted code will look like:
    // fn __fmt() {
    //     let output = ...;
    // }
    //
    // We want to extract just the "let output = ...;" part
    extract_statement_from_function(&formatted)
}

/// Extract the statement from a formatted function body
fn extract_statement_from_function(formatted: &str) -> String {
    // Find the opening brace and closing brace
    let start = formatted.find('{').map(|i| i + 1);
    let end = formatted.rfind('}');

    if let (Some(start), Some(end)) = (start, end) {
        let body = &formatted[start..end];
        // Trim whitespace and return just the statement
        body.trim().to_string()
    } else {
        // Fallback to original if parsing fails
        formatted.to_string()
    }
}
