//! Shared utilities for subgraph code generation in control flow nodes (If, Loop, Scan)

use super::prelude::*;
use onnx_ir::Node;
use std::collections::HashSet;

/// Generate outer-scope reference bindings for a subgraph.
///
/// Creates `let` bindings that map outer-scope values (from the parent graph)
/// to the names used within the subgraph.
///
/// # Parameters
/// - `outer_scope_inputs`: The node inputs that provide values for outer-scope references
/// - `scope_ref_names`: The original sanitized ONNX names that the subgraph uses
/// - `exclude_names`: Names to exclude from binding generation (e.g., loop-provided variables)
/// - `scope`: The parent scope for accessing outer values
/// - `node_position`: The position of the control flow node in the graph
pub(super) fn generate_outer_scope_bindings(
    outer_scope_inputs: &[Argument],
    scope_ref_names: &[String],
    exclude_names: &HashSet<String>,
    scope: &mut Scope,
    node_position: usize,
) -> TokenStream {
    let mut bindings = quote! {};

    for (idx, scope_ref_name) in scope_ref_names.iter().enumerate() {
        // Skip names that should be excluded (e.g., loop-provided variables)
        if exclude_names.contains(scope_ref_name) {
            continue;
        }

        if let Some(outer_input) = outer_scope_inputs.get(idx) {
            let var_name = quote::format_ident!("{}", scope_ref_name);
            let outer_var = scope.at_position(node_position).arg(outer_input);

            match &outer_input.ty {
                ArgType::Tensor(_) => {
                    bindings.extend(quote! {
                        let #var_name = #outer_var.clone();
                    });
                }
                ArgType::Scalar(_) => {
                    bindings.extend(quote! {
                        let #var_name = #outer_var;
                    });
                }
                _ => {}
            }
        }
    }

    bindings
}

/// Register subgraph inputs and build scope for generating node forward code.
///
/// This registers all subgraph tensors in the scope so they can be properly
/// referenced and cloned during code generation.
pub(super) fn register_subgraph_scope<PS: PrecisionSettings + 'static>(
    subgraph: &onnx_ir::OnnxGraph,
    scope: &mut Scope,
    node_position: usize,
) {
    // Register subgraph inputs in scope
    for input in &subgraph.inputs {
        if let ArgType::Tensor(_) = &input.ty {
            scope.tensor_register_variable(input, node_position);
        }
    }

    // Build scope for subgraph nodes: register outputs and future uses
    for (idx, node) in subgraph.nodes.iter().enumerate() {
        let subgraph_node_pos = node_position + idx + 1;

        // Register node outputs
        for output in <Node as NodeCodegen<PS>>::outputs(node) {
            if let ArgType::Tensor(_) = &output.ty {
                scope.tensor_register_variable(output, subgraph_node_pos);
            }
        }

        // Register future uses of node inputs.
        // We only track dynamic and constant arguments because:
        // - Dynamic: runtime values that need clone tracking for ownership
        // - Constant: values embedded in the model that may be referenced multiple times
        // - Static initializers are excluded because they're baked into the model at
        //   compile time and don't need runtime clone management
        for input in <Node as NodeCodegen<PS>>::inputs(node)
            .iter()
            .filter(|arg| arg.is_dynamic() || arg.is_constant())
        {
            if let ArgType::Tensor(_) = &input.ty {
                scope.tensor_register_future_use(input, subgraph_node_pos - 1);
            }
        }
    }

    // Register future uses for subgraph outputs
    for output in &subgraph.outputs {
        if let ArgType::Tensor(_) = &output.ty {
            scope.tensor_register_future_use(output, node_position + subgraph.nodes.len());
        }
    }
}

/// Generate forward code for all nodes in a subgraph.
pub(super) fn generate_subgraph_forward_code<PS: PrecisionSettings + 'static>(
    subgraph: &onnx_ir::OnnxGraph,
    scope: &mut Scope,
    node_position: usize,
) -> TokenStream {
    let mut code = quote! {};

    for (idx, node) in subgraph.nodes.iter().enumerate() {
        let mut scope_at_pos = scope.at_position(node_position + idx + 1);
        let node_code = <Node as NodeCodegen<PS>>::forward(node, &mut scope_at_pos);
        code.extend(node_code);
    }

    code
}
