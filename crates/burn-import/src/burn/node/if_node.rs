use super::prelude::*;
use super::subgraph_helper;
use std::collections::HashSet;

/// Generate inline code for a subgraph branch (then/else).
///
/// Returns (body_code, output_tuple)
fn generate_branch_code(
    subgraph: &onnx_ir::OnnxGraph,
    outer_scope_inputs: &[Argument],
    scope_ref_names: &[String],
    scope: &mut Scope,
    node_position: usize,
) -> (TokenStream, TokenStream) {
    // For If branches, all scope_ref_names are genuine outer-scope references
    // (no filtering needed, unlike Loop/Scan)
    let exclude_names = HashSet::new();

    // Collect names actually used in this branch to avoid unused variable warnings
    let used_names = subgraph_helper::collect_subgraph_referenced_names(subgraph);

    // Generate outer-scope bindings (only for names actually used in this branch)
    let bindings = subgraph_helper::generate_outer_scope_bindings(
        outer_scope_inputs,
        scope_ref_names,
        &exclude_names,
        Some(&used_names),
        scope,
        node_position,
    );

    // Register subgraph scope
    subgraph_helper::register_subgraph_scope(subgraph, scope, node_position);

    // Generate forward code
    let forward_code =
        subgraph_helper::generate_subgraph_forward_code(subgraph, scope, node_position);

    // Generate output tuple
    let output_names: Vec<_> = subgraph.outputs.iter().map(arg_to_ident).collect();
    let output_tuple = if output_names.len() == 1 {
        let out = &output_names[0];
        quote! { #out }
    } else {
        quote! { (#(#output_names),*) }
    };

    let body = quote! {
        #bindings
        #forward_code
    };

    (body, output_tuple)
}

impl NodeCodegen for onnx_ir::node::if_node::IfNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        // Get condition input (first input)
        let cond_arg = self
            .inputs
            .first()
            .expect("If node requires condition input");

        let cond = match &cond_arg.ty {
            ArgType::Scalar(_) => {
                let name = arg_to_ident(cond_arg);
                quote! { #name }
            }
            ArgType::Tensor(_) => {
                let cond_tensor = scope.arg(cond_arg);
                // Convert tensor to bool - assume it's a scalar tensor
                quote! { #cond_tensor.into_scalar().elem::<bool>() }
            }
            other => panic!("If condition must be scalar or tensor, got {:?}", other),
        };

        // Get outer-scope reference inputs (all inputs after condition)
        let outer_scope_inputs: Vec<_> = self.inputs.iter().skip(1).cloned().collect();

        // Generate code for then and else branches
        let node_position = scope.node_position();
        let (then_body, then_output) = generate_branch_code(
            &self.config.then_branch,
            &outer_scope_inputs,
            &self.config.scope_ref_names,
            scope.scope(),
            node_position,
        );
        let (else_body, else_output) = generate_branch_code(
            &self.config.else_branch,
            &outer_scope_inputs,
            &self.config.scope_ref_names,
            scope.scope(),
            node_position,
        );

        // Generate output variable declarations
        let output_names: Vec<_> = self.outputs.iter().map(arg_to_ident).collect();
        let output_decls = if self.outputs.len() == 1 {
            let out = &output_names[0];
            quote! { let #out }
        } else {
            quote! { let (#(#output_names),*) }
        };

        quote! {
            #output_decls = if #cond {
                #then_body
                #then_output
            } else {
                #else_body
                #else_output
            };
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        // Register imports from subgraph nodes
        let mut register_subgraph_imports = |subgraph: &onnx_ir::OnnxGraph| {
            for node in &subgraph.nodes {
                NodeCodegen::register_imports(node, imports);
            }
        };

        register_subgraph_imports(&self.config.then_branch);
        register_subgraph_imports(&self.config.else_branch);
    }
}

#[cfg(test)]
mod tests {
    // If node tests require complex OnnxGraph construction which is better tested
    // through integration tests
}
