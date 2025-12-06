use super::prelude::*;

/// Generate inline code for a loop body subgraph
fn generate_loop_body_code(
    subgraph: &onnx_ir::OnnxGraph,
    scope: &mut Scope,
    node_position: usize,
) -> TokenStream {
    let mut body = quote! {};

    // Register subgraph inputs in scope (they reference loop variables)
    for input in &subgraph.inputs {
        if let ArgType::Tensor(_) = &input.ty {
            scope.tensor_register_variable(input, node_position);
        }
    }

    // Build scope for subgraph nodes: register outputs and future uses
    for (idx, node) in subgraph.nodes.iter().enumerate() {
        let subgraph_node_pos = node_position + idx + 1;

        // Register node outputs
        for output in NodeCodegen::outputs(node) {
            if let ArgType::Tensor(_) = &output.ty {
                scope.tensor_register_variable(output, subgraph_node_pos);
            }
        }

        // Register future uses of node inputs
        // Filter to only dynamic/constant inputs (exclude static-only initializers)
        for input in NodeCodegen::inputs(node)
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

    // Generate forward code for each node
    for (idx, node) in subgraph.nodes.iter().enumerate() {
        let mut scope_at_pos = scope.at_position(node_position + idx + 1);
        let node_code = NodeCodegen::forward(node, &mut scope_at_pos);
        body.extend(node_code);
    }

    body
}

impl NodeCodegen for onnx_ir::node::loop_node::LoopNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        // Inputs: [M (max_trip_count), cond (initial condition), v_initial...]
        // Per ONNX spec, M and cond can be empty strings (optional)
        let max_trip_count_arg = &self.inputs[0];
        let init_cond_arg = &self.inputs[1];
        let v_initial_args: Vec<_> = self.inputs.iter().skip(2).collect();

        // Extract max trip count
        let max_count = if max_trip_count_arg.name.is_empty() {
            quote! { i64::MAX } // No limit if not provided
        } else {
            match &max_trip_count_arg.ty {
                ArgType::Scalar(_) => {
                    let name = arg_to_ident(max_trip_count_arg);
                    quote! { #name }
                }
                ArgType::Tensor(_) => {
                    let tensor = scope.arg(max_trip_count_arg);
                    quote! { #tensor.into_scalar().elem::<i64>() }
                }
                _ => panic!("Loop max_trip_count must be scalar i64"),
            }
        };

        // Extract initial condition
        let init_cond = if init_cond_arg.name.is_empty() {
            quote! { true } // Run if not provided
        } else {
            match &init_cond_arg.ty {
                ArgType::Scalar(_) => {
                    let name = arg_to_ident(init_cond_arg);
                    quote! { #name }
                }
                ArgType::Tensor(_) => {
                    let tensor = scope.arg(init_cond_arg);
                    quote! { #tensor.into_scalar().elem::<bool>() }
                }
                _ => panic!("Loop condition must be scalar bool"),
            }
        };

        // Body inputs: [iter_num, cond_in, v_in...]
        // Body outputs: [cond_out, v_out..., scan_outputs...]

        // Calculate number of loop-carried dependencies
        let num_loop_vars = v_initial_args.len();

        // Per ONNX spec, first N body outputs (after cond_out) are loop-carried deps
        // where N = number of v_initial inputs. Rest are scan outputs.
        let num_loop_carried_outputs = num_loop_vars;
        let num_scan_outputs = self.outputs.len() - num_loop_carried_outputs;

        // Get body input and output variable names
        // Body inputs: [iter_num, cond_in, v_in...]
        let iter_name = arg_to_ident(&self.config.body.inputs[0]);
        let cond_in_name = arg_to_ident(&self.config.body.inputs[1]);
        let loop_var_names: Vec<_> = self
            .config
            .body
            .inputs
            .iter()
            .skip(2) // Skip iter and cond_in
            .map(arg_to_ident)
            .collect();

        // Body outputs: [cond_out, v_out..., scan_outputs...]
        let cond_out_name = arg_to_ident(&self.config.body.outputs[0]);
        let loop_out_names: Vec<_> = self
            .config
            .body
            .outputs
            .iter()
            .skip(1)
            .take(num_loop_carried_outputs)
            .map(arg_to_ident)
            .collect();

        // Initialize loop-carried dependency variables
        // Only mark as mutable if the variable is actually updated (different name from output)
        let mut init_stmts = quote! {};
        for (idx, initial_arg) in v_initial_args.iter().enumerate() {
            let var_name = &loop_var_names[idx];
            let init_value = arg_to_ident(initial_arg);

            // Check if this variable will be updated (different name means it gets assigned)
            let needs_mut = idx < num_loop_carried_outputs
                && loop_out_names.get(idx).is_some_and(|out| var_name != out);

            if needs_mut {
                init_stmts.extend(quote! {
                    let mut #var_name = #init_value;
                });
            } else {
                init_stmts.extend(quote! {
                    let #var_name = #init_value;
                });
            }
        }

        // Initialize scan output collectors if any
        let mut scan_init = quote! {};
        let mut scan_collectors = vec![];
        let scan_out_args: Vec<_> = self
            .config
            .body
            .outputs
            .iter()
            .skip(1 + num_loop_vars)
            .collect();

        if num_scan_outputs > 0 {
            for i in 0..num_scan_outputs {
                let collector_name = syn::Ident::new(
                    &format!("scan_collector_{}", i),
                    proc_macro2::Span::call_site(),
                );
                scan_collectors.push(collector_name.clone());
                scan_init.extend(quote! {
                    let mut #collector_name = alloc::vec::Vec::new();
                });
            }
        }

        // Generate loop body code
        let node_position = scope.node_position();
        let body_code = generate_loop_body_code(&self.config.body, scope.scope(), node_position);

        // Update loop-carried variables after iteration
        // Skip self-assignments when body passes through a value unchanged (same name)
        let mut update_vars = quote! {};
        for (idx, out_name) in loop_out_names.iter().enumerate() {
            let var_name = &loop_var_names[idx];
            if var_name != out_name {
                update_vars.extend(quote! {
                    #var_name = #out_name;
                });
            }
        }

        // Update condition from body output (skip if same name)
        let update_cond = if cond_in_name != cond_out_name {
            quote! { #cond_in_name = #cond_out_name; }
        } else {
            quote! {}
        };

        // Collect scan outputs - handle scalar vs tensor
        let mut collect_scans = quote! {};
        for (idx, scan_arg) in scan_out_args.iter().enumerate() {
            let out_name = arg_to_ident(scan_arg);
            let collector = &scan_collectors[idx];

            // Tensors need to be cloned before collecting, scalars can be copied
            match &scan_arg.ty {
                ArgType::Scalar(_) => {
                    collect_scans.extend(quote! {
                        #collector.push(#out_name);
                    });
                }
                ArgType::Tensor(_) => {
                    collect_scans.extend(quote! {
                        #collector.push(#out_name.clone());
                    });
                }
                _ => panic!("Scan output must be scalar or tensor"),
            }
        }

        // Build output tuple: (loop-carried values, concatenated scan outputs)
        let output_names: Vec<_> = self.outputs.iter().map(arg_to_ident).collect();

        // Collect final values for outputs
        let mut output_values = vec![];

        // First outputs are final loop-carried dependencies
        for (idx, _) in output_names
            .iter()
            .take(num_loop_carried_outputs)
            .enumerate()
        {
            let var_name = &loop_var_names[idx];
            output_values.push(quote! { #var_name });
        }

        // Remaining outputs are concatenated scan outputs
        for (idx, scan_arg) in scan_out_args.iter().enumerate() {
            let collector = &scan_collectors[idx];

            // Handle scalar vs tensor scan outputs
            match &scan_arg.ty {
                ArgType::Scalar(_) => {
                    // Convert Vec<scalar> to 2D tensor with shape [N, 1]
                    // ONNX spec: scan outputs from scalars get an added dimension
                    output_values.push(quote! {
                        {
                            let data = TensorData::from(#collector.as_slice());
                            let len = #collector.len();
                            let tensor1d: Tensor<B, 1> = Tensor::from_data(data, &*self.device);
                            tensor1d.reshape([len, 1])
                        }
                    });
                }
                ArgType::Tensor(_) => {
                    // Concatenate tensors
                    output_values.push(quote! { Tensor::cat(#collector, 0) });
                }
                _ => panic!("Scan output must be scalar or tensor"),
            }
        }

        // Generate output declaration (let outputs = ...)
        let output_decls = if self.outputs.len() == 1 {
            let out = &output_names[0];
            quote! { let #out }
        } else {
            quote! { let (#(#output_names),*) }
        };

        // Generate output tuple
        let output_tuple = if self.outputs.len() == 1 {
            let val = &output_values[0];
            quote! { #val }
        } else {
            quote! { (#(#output_values),*) }
        };

        // cond_in only needs mut if it's actually updated (different name from output)
        let cond_needs_mut = cond_in_name != cond_out_name;
        let cond_init = if cond_needs_mut {
            quote! { let mut #cond_in_name = #init_cond; }
        } else {
            quote! { let #cond_in_name = #init_cond; }
        };

        quote! {
            #[allow(unused_variables, unused_assignments)]
            #output_decls = {
                #init_stmts
                #scan_init

                let mut #iter_name = 0_i64;
                #cond_init

                while #cond_in_name && #iter_name < #max_count {
                    #body_code

                    // Collect scan outputs from body outputs (before updating variables)
                    #collect_scans

                    // Update loop-carried variables for next iteration
                    #update_vars

                    // Update condition from body output for next iteration
                    #update_cond
                    #iter_name += 1;
                }

                #output_tuple
            };
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        // Register imports from subgraph nodes
        for node in &self.config.body.nodes {
            NodeCodegen::register_imports(node, imports);
        }

        let num_loop_vars = self.inputs.len() - 2; // Subtract M and cond
        let num_scan_outputs = self.outputs.len() - num_loop_vars;

        // Register Tensor for scan outputs if needed
        if num_scan_outputs > 0 {
            imports.register("burn::tensor::Tensor");

            // Check if any scan outputs are scalars - need TensorData import
            let scan_out_args: Vec<_> = self
                .config
                .body
                .outputs
                .iter()
                .skip(1 + num_loop_vars)
                .collect();

            for scan_arg in scan_out_args {
                if matches!(&scan_arg.ty, ArgType::Scalar(_)) {
                    imports.register("burn::tensor::TensorData");
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    // Loop node tests require complex OnnxGraph construction which is better tested
    // through integration tests
}
