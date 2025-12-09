use super::prelude::*;
use super::subgraph_helper;
use std::collections::HashSet;

/// Generate inline code for a scan body subgraph.
///
/// Scan body inputs (state variables and scan input elements) are excluded from
/// outer-scope bindings since they're provided by the scan construct.
fn generate_scan_body_code(
    subgraph: &onnx_ir::OnnxGraph,
    outer_scope_inputs: &[Argument],
    scope_ref_names: &[String],
    body_input_names: &HashSet<String>,
    scope: &mut Scope,
    node_position: usize,
) -> TokenStream {
    // Collect names actually used in this body to avoid unused variable warnings
    let used_names = subgraph_helper::collect_subgraph_referenced_names(subgraph);

    // Generate outer-scope bindings (excluding scan-provided body inputs, only for used names)
    let bindings = subgraph_helper::generate_outer_scope_bindings(
        outer_scope_inputs,
        scope_ref_names,
        body_input_names,
        Some(&used_names),
        scope,
        node_position,
    );

    // Register subgraph scope
    subgraph_helper::register_subgraph_scope(subgraph, scope, node_position);

    // Generate forward code
    let forward_code =
        subgraph_helper::generate_subgraph_forward_code(subgraph, scope, node_position);

    quote! {
        #bindings
        #forward_code
    }
}

impl NodeCodegen for onnx_ir::node::scan_node::ScanNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let num_scan_inputs = self.config.num_scan_inputs as usize;

        // Calculate how many outer-scope refs were added (beyond ONNX inputs)
        let num_outer_scope_refs = self.config.scope_ref_names.len();
        let num_onnx_inputs = self.inputs.len() - num_outer_scope_refs;
        let num_state_vars = num_onnx_inputs - num_scan_inputs;

        // Outer-scope references (values from parent scope that subgraph needs)
        let outer_scope_inputs: Vec<_> =
            self.inputs.iter().skip(num_onnx_inputs).cloned().collect();

        // Split ONNX inputs into state variables and scan input sequences
        let initial_state_vars: Vec<_> = self.inputs.iter().take(num_state_vars).collect();
        let scan_input_sequences: Vec<_> = self
            .inputs
            .iter()
            .skip(num_state_vars)
            .take(num_scan_inputs)
            .collect();

        // Body inputs: [state_vars..., scan_inputs...]
        // Body outputs: [state_vars_out..., scan_outputs...]
        let num_scan_outputs = self.config.body.outputs.len() - num_state_vars;

        // Get sequence length from first scan input
        let first_scan_input = scan_input_sequences[0];
        let scan_axis = self.config.scan_input_axes.first().copied().unwrap_or(0) as usize;
        let first_scan_name = arg_to_ident(first_scan_input);
        let seq_len_expr = quote! { #first_scan_name.shape().dims[#scan_axis] };

        // Initialize state variables
        let mut init_stmts = quote! {};
        let state_var_names: Vec<_> = self
            .config
            .body
            .inputs
            .iter()
            .take(num_state_vars)
            .map(arg_to_ident)
            .collect();

        for (idx, initial_arg) in initial_state_vars.iter().enumerate() {
            let var_name = &state_var_names[idx];
            let init_value = arg_to_ident(initial_arg);
            init_stmts.extend(quote! {
                let mut #var_name = #init_value;
            });
        }

        // Initialize scan output accumulators
        let scan_output_collectors: Vec<_> = (0..num_scan_outputs)
            .map(|i| {
                syn::Ident::new(
                    &format!("scan_output_collector_{}", i),
                    proc_macro2::Span::call_site(),
                )
            })
            .collect();

        for collector in &scan_output_collectors {
            init_stmts.extend(quote! {
                let mut #collector = alloc::vec::Vec::new();
            });
        }

        init_stmts.extend(quote! {
            let seq_len = #seq_len_expr;
        });

        // Extract scan input elements for current iteration
        let scan_input_names: Vec<_> = self
            .config
            .body
            .inputs
            .iter()
            .skip(num_state_vars)
            .map(arg_to_ident)
            .collect();

        let mut slice_stmts = quote! {};
        for (idx, scan_input_arg) in scan_input_sequences.iter().enumerate() {
            let var_name = &scan_input_names[idx];
            let tensor_name = arg_to_ident(scan_input_arg);

            let reverse = self
                .config
                .scan_input_directions
                .get(idx)
                .copied()
                .unwrap_or(0)
                == 1;
            let scan_axis = self.config.scan_input_axes.get(idx).copied().unwrap_or(0) as usize;

            let iter_expr = if reverse {
                quote! { seq_len - i - 1 }
            } else {
                quote! { i }
            };

            // Get rank from tensor type
            let ArgType::Tensor(tensor_ty) = &scan_input_arg.ty else {
                panic!("Scan input must be tensor");
            };
            let input_rank = tensor_ty.rank;
            let output_rank = input_rank - 1;

            // Slice along scan axis and squeeze to remove that dimension
            slice_stmts.extend(quote! {
                let #var_name = #tensor_name
                    .clone()
                    .slice_dim(#scan_axis, #iter_expr..#iter_expr + 1)
                    .squeeze_dim::<#output_rank>(#scan_axis);
            });
        }

        // Collect body input names (state variables + scan input elements)
        // These should NOT be treated as outer-scope references even though
        // they're declared as subgraph inputs without initializers
        let body_input_names: HashSet<String> = self
            .config
            .body
            .inputs
            .iter()
            .map(|arg| arg.name.clone())
            .collect();

        // Generate body code
        let node_position = scope.node_position();
        let body_code = generate_scan_body_code(
            &self.config.body,
            &outer_scope_inputs,
            &self.config.scope_ref_names,
            &body_input_names,
            scope.scope(),
            node_position,
        );

        // Update state variables and collect scan outputs
        let mut update_stmts = quote! {};

        // Update state variables (clone if there are scan outputs)
        let should_clone = num_scan_outputs > 0;
        for (idx, var_name) in state_var_names.iter().enumerate() {
            let out_name = arg_to_ident(&self.config.body.outputs[idx]);
            if should_clone {
                update_stmts.extend(quote! {
                    #var_name = #out_name.clone();
                });
            } else {
                update_stmts.extend(quote! {
                    #var_name = #out_name;
                });
            }
        }

        // Collect scan outputs
        for (idx, collector) in scan_output_collectors.iter().enumerate() {
            let body_out_name = arg_to_ident(&self.config.body.outputs[num_state_vars + idx]);
            update_stmts.extend(quote! {
                #collector.push(#body_out_name);
            });
        }

        // Finalize scan outputs: reverse if needed, then stack into tensors
        let mut finalize_stmts = quote! {};
        let final_scan_output_names: Vec<_> = (0..num_scan_outputs)
            .map(|i| {
                syn::Ident::new(
                    &format!("scan_output_{}", i),
                    proc_macro2::Span::call_site(),
                )
            })
            .collect();

        for (idx, (collector, output_name)) in scan_output_collectors
            .iter()
            .zip(final_scan_output_names.iter())
            .enumerate()
        {
            let reverse = self
                .config
                .scan_output_directions
                .get(idx)
                .copied()
                .unwrap_or(0)
                == 1;
            if reverse {
                finalize_stmts.extend(quote! {
                    #collector.reverse();
                });
            }

            let stack_axis = self.config.scan_output_axes.get(idx).copied().unwrap_or(0) as usize;
            finalize_stmts.extend(quote! {
                let #output_name = Tensor::stack(#collector.clone(), #stack_axis);
            });
        }

        // Collect all output names (final state vars + scan outputs)
        let all_output_names: Vec<_> = self.outputs.iter().map(arg_to_ident).collect();

        // Map to actual variable names
        let output_vars: Vec<_> = (0..self.outputs.len())
            .map(|idx| {
                if idx < num_state_vars {
                    state_var_names[idx].clone()
                } else {
                    final_scan_output_names[idx - num_state_vars].clone()
                }
            })
            .collect();

        if self.outputs.len() == 1 {
            let output_name = &all_output_names[0];
            let var_name = &output_vars[0];

            quote! {
                #[allow(unused_variables, unused_assignments)]
                let #output_name = {
                    #init_stmts

                    for i in 0..seq_len {
                        #slice_stmts
                        #body_code
                        #update_stmts
                    }

                    #finalize_stmts
                    #var_name
                };
            }
        } else {
            quote! {
                #[allow(unused_variables, unused_assignments)]
                let (#(#all_output_names),*) = {
                    #init_stmts

                    for i in 0..seq_len {
                        #slice_stmts
                        #body_code
                        #update_stmts
                    }

                    #finalize_stmts
                    (#(#output_vars),*)
                };
            }
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        // Register imports from body nodes
        for node in &self.config.body.nodes {
            NodeCodegen::register_imports(node, imports);
        }
    }
}

#[cfg(test)]
mod tests {
    // Scan node tests require complex OnnxGraph construction which is better tested
    // through integration tests
}
