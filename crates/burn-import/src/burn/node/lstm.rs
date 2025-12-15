//! ONNX LSTM node import implementation.
//!
//! ## Supported ONNX Features
//!
//! - Forward, reverse, and bidirectional directions
//! - Batch-first and sequence-first layouts (`layout` attribute)
//! - Initial hidden and cell states
//! - Custom activations: Sigmoid, Tanh, Relu, HardSigmoid, LeakyRelu, Softplus
//! - Cell state clipping (`clip` attribute)
//! - Input-forget gate coupling (`input_forget` attribute)
//!
//! ## Unsupported ONNX Features
//!
//! - **Peephole connections**: ONNX input `P` with shape `[num_directions, 3*hidden_size]` allows
//!   gates to "peek" at the cell state. This is rarely used in modern models.
//!
//! - **Variable sequence lengths**: ONNX input `sequence_lens` with shape `[batch_size]` specifies
//!   the actual length of each sequence in a batch. Currently, all sequences in a batch must have
//!   the same length.

use super::prelude::*;
use burn::nn::activation::ActivationConfig;
use burn_store::TensorSnapshot;
use onnx_ir::lstm::{LstmActivationFunction, LstmDirection};

/// Convert ONNX activation function to Burn ActivationConfig.
///
/// # Panics
///
/// Panics if the ONNX activation function is not supported by burn-nn.
/// Supported activations: Sigmoid, Tanh, Relu, HardSigmoid, LeakyRelu, Softplus.
fn to_burn_activation(onnx_activation: LstmActivationFunction) -> ActivationConfig {
    match onnx_activation {
        LstmActivationFunction::Sigmoid => ActivationConfig::Sigmoid,
        LstmActivationFunction::Tanh => ActivationConfig::Tanh,
        LstmActivationFunction::Relu => ActivationConfig::Relu,
        LstmActivationFunction::HardSigmoid => {
            ActivationConfig::HardSigmoid(burn::nn::HardSigmoidConfig::new())
        }
        LstmActivationFunction::LeakyRelu => {
            ActivationConfig::LeakyRelu(burn::nn::LeakyReluConfig::new())
        }
        LstmActivationFunction::Softplus => {
            ActivationConfig::Softplus(burn::nn::SoftplusConfig::new())
        }
        unsupported => panic!(
            "LSTM activation '{:?}' is not supported by burn-nn. \
             Supported activations: Sigmoid, Tanh, Relu, HardSigmoid, LeakyRelu, Softplus. \
             Consider using a supported activation or implementing support in burn-nn.",
            unsupported
        ),
    }
}

/// Collect tensor snapshots for LSTM burnpack serialization.
///
/// This function handles the complex weight transformation from ONNX's packed format
/// to Burn's individual GateController structure using NdArray backend for tensor ops.
///
/// ONNX LSTM weight layout:
/// - W: `[num_directions, 4*hidden_size, input_size]` - gates ordered as [i, o, f, c]
/// - R: `[num_directions, 4*hidden_size, hidden_size]` - gates ordered as [i, o, f, c]
/// - B: `[num_directions, 8*hidden_size]` - Wb[i,o,f,c] then Rb[i,o,f,c]
///
/// Burn LSTM structure (per direction):
/// - input_gate.input_transform: weight `[input_size, hidden_size]`, bias `[hidden_size]`
/// - input_gate.hidden_transform: weight `[hidden_size, hidden_size]`, bias `[hidden_size]`
/// - forget_gate, output_gate, cell_gate: same structure
#[allow(clippy::single_range_in_vec_init)]
fn collect_lstm_snapshots(
    field_name: &str,
    inputs: &[Argument],
    config: &onnx_ir::lstm::LstmConfig,
) -> Vec<TensorSnapshot> {
    use crate::burn::node_traits::{SerializationBackend, extract_node_data};
    use burn::tensor::Tensor;

    let hidden_size = config.hidden_size;
    let input_size = config.input_size;

    // Extract weight tensors from inputs
    let data_w = extract_node_data(inputs, 1);
    let data_r = extract_node_data(inputs, 2);
    let data_b = extract_node_data(inputs, 3);

    let Some(data_w) = data_w else {
        return vec![];
    };
    let Some(data_r) = data_r else {
        return vec![];
    };

    let dtype = data_w.dtype;
    let device = Default::default();

    // ONNX gate order: i(input), o(output), f(forget), c(cell)
    // Burn gate order: input_gate, forget_gate, output_gate, cell_gate
    let onnx_to_burn_gate_order = [0usize, 2, 1, 3]; // input, forget, output, cell
    let gate_names = ["input_gate", "forget_gate", "output_gate", "cell_gate"];

    // Determine direction prefixes based on LSTM type
    let direction_prefixes: Vec<&str> = match config.direction {
        LstmDirection::Forward | LstmDirection::Reverse => vec![""],
        LstmDirection::Bidirectional => vec!["forward.", "reverse."],
    };

    let mut snapshots = Vec::new();

    // Create tensors from data
    let w_tensor: Tensor<SerializationBackend, 3> = Tensor::from_data(data_w.clone(), &device);
    let r_tensor: Tensor<SerializationBackend, 3> = Tensor::from_data(data_r.clone(), &device);
    let b_tensor: Option<Tensor<SerializationBackend, 2>> =
        data_b.clone().map(|b| Tensor::from_data(b, &device));

    for (dir_idx, dir_prefix) in direction_prefixes.iter().enumerate() {
        // Select direction slice from W and R
        // W shape: [num_directions, 4*hidden_size, input_size]
        let w_dir = w_tensor
            .clone()
            .slice([dir_idx..dir_idx + 1, 0..4 * hidden_size, 0..input_size])
            .squeeze::<2>(); // [4*hidden_size, input_size]

        // R shape: [num_directions, 4*hidden_size, hidden_size]
        let r_dir = r_tensor
            .clone()
            .slice([dir_idx..dir_idx + 1, 0..4 * hidden_size, 0..hidden_size])
            .squeeze::<2>(); // [4*hidden_size, hidden_size]

        // B shape: [num_directions, 8*hidden_size]
        let b_dir = b_tensor.as_ref().map(|b| {
            b.clone()
                .slice([dir_idx..dir_idx + 1, 0..8 * hidden_size])
                .squeeze::<1>() // [8*hidden_size]
        });

        for (gate_idx, gate_name) in gate_names.iter().enumerate() {
            let onnx_gate_idx = onnx_to_burn_gate_order[gate_idx];
            let start = onnx_gate_idx * hidden_size;
            let end = start + hidden_size;

            // Input transform weight: slice from W and transpose
            // ONNX: [hidden_size, input_size] -> Burn: [input_size, hidden_size]
            let w_gate = w_dir.clone().slice([start..end, 0..input_size]).transpose(); // [input_size, hidden_size]
            let w_gate_data = w_gate.into_data();

            let path = format!(
                "{}.{}{}.input_transform.weight",
                field_name, dir_prefix, gate_name
            );
            snapshots.push(create_snapshot_from_data(
                w_gate_data,
                &path,
                "Linear",
                dtype,
            ));

            // Input transform bias: Wb + Rb for this gate
            if let Some(ref b) = b_dir {
                let wb_start = onnx_gate_idx * hidden_size;
                let wb_end = wb_start + hidden_size;
                let rb_start = 4 * hidden_size + onnx_gate_idx * hidden_size;
                let rb_end = rb_start + hidden_size;

                let wb: Tensor<SerializationBackend, 1> = b.clone().slice([wb_start..wb_end]);
                let rb: Tensor<SerializationBackend, 1> = b.clone().slice([rb_start..rb_end]);
                let bias = wb.add(rb);
                let bias_data = bias.into_data();

                let path = format!(
                    "{}.{}{}.input_transform.bias",
                    field_name, dir_prefix, gate_name
                );
                snapshots.push(create_snapshot_from_data(bias_data, &path, "Linear", dtype));
            }

            // Hidden transform weight: slice from R and transpose
            // ONNX: [hidden_size, hidden_size] -> Burn: [hidden_size, hidden_size]
            let r_gate = r_dir
                .clone()
                .slice([start..end, 0..hidden_size])
                .transpose(); // [hidden_size, hidden_size]
            let r_gate_data = r_gate.into_data();

            let path = format!(
                "{}.{}{}.hidden_transform.weight",
                field_name, dir_prefix, gate_name
            );
            snapshots.push(create_snapshot_from_data(
                r_gate_data,
                &path,
                "Linear",
                dtype,
            ));

            // Hidden transform bias: zeros (combined bias is in input_transform)
            if b_dir.is_some() {
                let zeros: Tensor<SerializationBackend, 1> = Tensor::zeros([hidden_size], &device);
                let zeros_data = zeros.into_data();

                let path = format!(
                    "{}.{}{}.hidden_transform.bias",
                    field_name, dir_prefix, gate_name
                );
                snapshots.push(create_snapshot_from_data(
                    zeros_data, &path, "Linear", dtype,
                ));
            }
        }
    }

    snapshots
}

/// Create a TensorSnapshot from TensorData.
///
/// Converts the data to the target dtype to preserve the original precision.
/// This is important because intermediate tensor operations use f64 for precision,
/// but we need to store the data in the original dtype (e.g., F16, F32).
fn create_snapshot_from_data(
    data: burn::tensor::TensorData,
    path: &str,
    container_type: &str,
    dtype: burn::tensor::DType,
) -> TensorSnapshot {
    use burn::module::ParamId;
    use burn_store::TensorSnapshotError;
    use std::rc::Rc;

    // Convert data back to the original dtype
    // This is necessary because we use f64 for intermediate operations to preserve precision
    let data = data.convert_dtype(dtype);

    let shape = data.shape.clone();
    let path_stack: Vec<String> = path.split('.').map(String::from).collect();
    let container_stack = vec![format!("Struct:{}", container_type)];

    let data_fn = Rc::new(
        move || -> Result<burn::tensor::TensorData, TensorSnapshotError> { Ok(data.clone()) },
    );

    TensorSnapshot::from_closure(
        data_fn,
        dtype,
        shape,
        path_stack,
        container_stack,
        ParamId::new(),
    )
}

/// Convert ActivationConfig to tokens for code generation
fn activation_to_tokens(activation: &ActivationConfig) -> TokenStream {
    match activation {
        ActivationConfig::Sigmoid => quote! { ActivationConfig::Sigmoid },
        ActivationConfig::Tanh => quote! { ActivationConfig::Tanh },
        ActivationConfig::Relu => quote! { ActivationConfig::Relu },
        ActivationConfig::HardSigmoid(_) => {
            quote! { ActivationConfig::HardSigmoid(burn::nn::HardSigmoidConfig::new()) }
        }
        ActivationConfig::LeakyRelu(_) => {
            quote! { ActivationConfig::LeakyRelu(burn::nn::LeakyReluConfig::new()) }
        }
        ActivationConfig::Softplus(_) => {
            quote! { ActivationConfig::Softplus(burn::nn::SoftplusConfig::new()) }
        }
        _ => panic!("Unsupported activation config for LSTM"),
    }
}

impl NodeCodegen for onnx_ir::lstm::LstmNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        let name = Ident::new(&self.name, Span::call_site());
        let d_input = self.config.input_size.to_tokens();
        let d_hidden = self.config.hidden_size.to_tokens();
        let bias = self.config.has_bias;
        let batch_first = self.config.batch_first;
        let input_forget = self.config.input_forget;

        // Convert activations to tokens
        let gate_act = to_burn_activation(self.config.gate_activation);
        let cell_act = to_burn_activation(self.config.cell_activation);
        let hidden_act = to_burn_activation(self.config.hidden_activation);

        let gate_activation = activation_to_tokens(&gate_act);
        let cell_activation = activation_to_tokens(&cell_act);
        let hidden_activation = activation_to_tokens(&hidden_act);

        // Generate clip config if present
        let clip_config = if let Some(clip) = self.config.clip {
            let clip_val = clip as f64;
            quote! { .with_clip(Some(#clip_val)) }
        } else {
            quote! {}
        };

        // Only add non-default activations to config
        let activations_config = {
            let mut tokens = quote! {};
            if !matches!(gate_act, ActivationConfig::Sigmoid) {
                tokens = quote! { #tokens .with_gate_activation(#gate_activation) };
            }
            if !matches!(cell_act, ActivationConfig::Tanh) {
                tokens = quote! { #tokens .with_cell_activation(#cell_activation) };
            }
            if !matches!(hidden_act, ActivationConfig::Tanh) {
                tokens = quote! { #tokens .with_hidden_activation(#hidden_activation) };
            }
            tokens
        };

        match self.config.direction {
            LstmDirection::Forward => Some(Field::new(
                self.name.clone(),
                quote! { Lstm<B> },
                quote! {
                    let #name = LstmConfig::new(#d_input, #d_hidden, #bias)
                        .with_batch_first(#batch_first)
                        .with_input_forget(#input_forget)
                        #clip_config
                        #activations_config
                        .init(device);
                },
            )),
            LstmDirection::Reverse => Some(Field::new(
                self.name.clone(),
                quote! { Lstm<B> },
                quote! {
                    let #name = LstmConfig::new(#d_input, #d_hidden, #bias)
                        .with_batch_first(#batch_first)
                        .with_reverse(true)
                        .with_input_forget(#input_forget)
                        #clip_config
                        #activations_config
                        .init(device);
                },
            )),
            LstmDirection::Bidirectional => Some(Field::new(
                self.name.clone(),
                quote! { BiLstm<B> },
                quote! {
                    let #name = BiLstmConfig::new(#d_input, #d_hidden, #bias)
                        .with_batch_first(#batch_first)
                        .with_input_forget(#input_forget)
                        #clip_config
                        #activations_config
                        .init(device);
                },
            )),
        }
    }

    fn collect_snapshots(&self, field_name: &str) -> Vec<TensorSnapshot> {
        collect_lstm_snapshots(field_name, &self.inputs, &self.config)
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let field = Ident::new(&self.name, Span::call_site());

        // Get output variable names
        let output_y = self.outputs.first().map(arg_to_ident);
        let output_y_h = self.outputs.get(1).map(arg_to_ident);
        let output_y_c = self.outputs.get(2).map(arg_to_ident);

        // Handle initial states if provided
        let has_initial_h = self.config.has_initial_h;
        let has_initial_c = self.config.has_initial_c;

        // Get initial state inputs if present
        // Input indices: 0=X, 1=W, 2=R, 3=B, 4=sequence_lens, 5=initial_h, 6=initial_c
        // ONNX initial states: [num_directions, batch_size, hidden_size]
        // Burn expects: [batch_size, hidden_size] for unidirectional
        let initial_state_expr = if has_initial_h && has_initial_c {
            let h_input = scope.arg(&self.inputs[5]);
            let c_input = scope.arg(&self.inputs[6]);
            match self.config.direction {
                LstmDirection::Forward | LstmDirection::Reverse => {
                    // Squeeze out the direction dimension (index 0) for unidirectional LSTM
                    // ONNX: [1, batch_size, hidden_size] -> Burn: [batch_size, hidden_size]
                    quote! { Some(LstmState::new(#c_input.squeeze_dim(0), #h_input.squeeze_dim(0))) }
                }
                LstmDirection::Bidirectional => {
                    // For bidirectional, keep all dimensions but reshape appropriately
                    quote! { Some(LstmState::new(#c_input, #h_input)) }
                }
            }
        } else {
            quote! { None }
        };

        // The LSTM module now handles batch_first and reverse internally via config,
        // so no input/output transformation is needed here
        let forward_call = quote! {
            let (output_seq, final_state) = self.#field.forward(#input, #initial_state_expr);
        };

        // Transform outputs to ONNX format
        // Burn output shape depends on batch_first config:
        //   batch_first=true:  [batch_size, seq_length, hidden_size] or [batch_size, seq_length, 2*hidden_size] for bidirectional
        //   batch_first=false: [seq_length, batch_size, hidden_size] or [seq_length, batch_size, 2*hidden_size] for bidirectional
        // ONNX Y output: [seq_length, num_directions, batch_size, hidden_size]
        // Y_h: [num_directions, batch_size, hidden_size]
        // Y_c: [num_directions, batch_size, hidden_size]

        // For unidirectional LSTM:
        //   - Burn final_state.hidden/cell: [batch_size, hidden_size] (2D)
        //   - Need to unsqueeze to add num_directions dimension
        //   - Burn output: [seq, batch, hidden] -> ONNX Y: [seq, 1, batch, hidden]
        // For bidirectional LSTM:
        //   - Burn final_state.hidden/cell: [2, batch_size, hidden_size] (already 3D)
        //   - No unsqueeze needed
        //   - Burn output: [seq, batch, 2*hidden] -> ONNX Y: [seq, 2, batch, hidden]
        //     This requires reshape + transpose
        let is_bidirectional = matches!(self.config.direction, LstmDirection::Bidirectional);
        let hidden_size = self.config.hidden_size;

        let (hidden_expr, cell_expr) = if is_bidirectional {
            (quote! { final_state.hidden }, quote! { final_state.cell })
        } else {
            (
                quote! { final_state.hidden.unsqueeze_dims::<3>(&[0]) },
                quote! { final_state.cell.unsqueeze_dims::<3>(&[0]) },
            )
        };

        // Y output transformation
        // For unidirectional: unsqueeze at dim 1 to add num_directions=1
        // For bidirectional: reshape to split the concatenated hidden states, then reorder dims
        //   ONNX layout=0 (batch_first=false): Y is [seq, num_dirs, batch, hidden]
        //   ONNX layout=1 (batch_first=true):  Y is [batch, seq, num_dirs, hidden]
        let y_output_expr = if is_bidirectional {
            let batch_first = self.config.batch_first;
            if batch_first {
                // Burn output: [batch, seq, 2*hidden]
                // Reshape to: [batch, seq, 2, hidden] - already matches ONNX layout=1
                quote! {
                    {
                        let [batch_size, seq_len, _] = output_seq.dims();
                        output_seq.reshape([batch_size, seq_len, 2, #hidden_size])
                    }
                }
            } else {
                // Burn output: [seq, batch, 2*hidden]
                // Reshape to: [seq, batch, 2, hidden]
                // Then swap dims 1 and 2 to get: [seq, 2, batch, hidden] for ONNX layout=0
                quote! {
                    {
                        let [seq_len, batch_size, _] = output_seq.dims();
                        let reshaped = output_seq.reshape([seq_len, batch_size, 2, #hidden_size]);
                        reshaped.swap_dims(1, 2)
                    }
                }
            }
        } else {
            quote! { output_seq.unsqueeze_dims::<4>(&[1]) }
        };

        // Build output assignments based on which outputs are used
        // Use block scoping to contain temporary variables
        match (output_y, output_y_h, output_y_c) {
            (Some(y), Some(y_h), Some(y_c)) => {
                quote! {
                    let (#y, #y_h, #y_c) = {
                        #forward_call
                        (
                            #y_output_expr,
                            #hidden_expr,
                            #cell_expr
                        )
                    };
                }
            }
            (Some(y), Some(y_h), None) => {
                quote! {
                    let (#y, #y_h) = {
                        #forward_call
                        (
                            #y_output_expr,
                            #hidden_expr
                        )
                    };
                }
            }
            (Some(y), None, None) => {
                quote! {
                    let #y = {
                        #forward_call
                        #y_output_expr
                    };
                }
            }
            (None, Some(y_h), Some(y_c)) => {
                quote! {
                    let (#y_h, #y_c) = {
                        #forward_call
                        (
                            #hidden_expr,
                            #cell_expr
                        )
                    };
                }
            }
            _ => {
                // Handle remaining cases - just run the forward pass
                quote! {
                    {
                        #forward_call
                    }
                }
            }
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        // Check if we need to import ActivationConfig (for non-default activations)
        let gate_act = to_burn_activation(self.config.gate_activation);
        let cell_act = to_burn_activation(self.config.cell_activation);
        let hidden_act = to_burn_activation(self.config.hidden_activation);

        let needs_activation_import = !matches!(gate_act, ActivationConfig::Sigmoid)
            || !matches!(cell_act, ActivationConfig::Tanh)
            || !matches!(hidden_act, ActivationConfig::Tanh);

        if needs_activation_import {
            imports.register("burn::nn::ActivationConfig");
        }

        match self.config.direction {
            LstmDirection::Forward | LstmDirection::Reverse => {
                imports.register("burn::nn::Lstm");
                imports.register("burn::nn::LstmConfig");
                imports.register("burn::nn::LstmState");
            }
            LstmDirection::Bidirectional => {
                imports.register("burn::nn::BiLstm");
                imports.register("burn::nn::BiLstmConfig");
                imports.register("burn::nn::LstmState");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::ir::{ArgType, Argument, TensorType};
    use onnx_ir::lstm::{LstmActivationFunction, LstmConfig, LstmDirection, LstmNode};

    fn create_lstm_node(
        name: &str,
        direction: LstmDirection,
        batch_first: bool,
        num_outputs: usize,
    ) -> LstmNode {
        let config = LstmConfig::new(
            4, // input_size
            8, // hidden_size
            direction,
            true,  // has_bias
            false, // has_initial_h
            false, // has_initial_c
            false, // has_peephole
            batch_first,
            None,                            // clip
            false,                           // input_forget
            LstmActivationFunction::Sigmoid, // gate_activation
            LstmActivationFunction::Tanh,    // cell_activation
            LstmActivationFunction::Tanh,    // hidden_activation
        );

        let input = Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::F32, 3, None)),
        );
        let w = Argument::new("W", ArgType::Tensor(TensorType::new(DType::F32, 3, None)));
        let r = Argument::new("R", ArgType::Tensor(TensorType::new(DType::F32, 3, None)));
        let b = Argument::new("B", ArgType::Tensor(TensorType::new(DType::F32, 2, None)));

        let mut outputs = vec![];
        if num_outputs > 0 {
            outputs.push(Argument::new(
                "Y",
                ArgType::Tensor(TensorType::new(DType::F32, 4, None)),
            ));
        }
        if num_outputs > 1 {
            outputs.push(Argument::new(
                "Y_h",
                ArgType::Tensor(TensorType::new(DType::F32, 3, None)),
            ));
        }
        if num_outputs > 2 {
            outputs.push(Argument::new(
                "Y_c",
                ArgType::Tensor(TensorType::new(DType::F32, 3, None)),
            ));
        }

        LstmNode {
            name: name.to_string(),
            inputs: vec![input, w, r, b],
            outputs,
            config,
        }
    }

    #[test]
    fn test_lstm_forward_basic() {
        let node = create_lstm_node("lstm1", LstmDirection::Forward, false, 3);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<B, 3>,
            W: Tensor<B, 3>,
            R: Tensor<B, 3>,
            B: Tensor<B, 2>,
        ) -> (Tensor<B, 4>, Tensor<B, 3>, Tensor<B, 3>) {
            let (Y, Y_h, Y_c) = {
                let (output_seq, final_state) = self.lstm1.forward(input, None);
                (
                    output_seq.unsqueeze_dims::<4>(&[1]),
                    final_state.hidden.unsqueeze_dims::<3>(&[0]),
                    final_state.cell.unsqueeze_dims::<3>(&[0]),
                )
            };
            (Y, Y_h, Y_c)
        }
        ");
    }

    #[test]
    fn test_lstm_forward_bidirectional() {
        let node = create_lstm_node("lstm1", LstmDirection::Bidirectional, false, 3);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<B, 3>,
            W: Tensor<B, 3>,
            R: Tensor<B, 3>,
            B: Tensor<B, 2>,
        ) -> (Tensor<B, 4>, Tensor<B, 3>, Tensor<B, 3>) {
            let (Y, Y_h, Y_c) = {
                let (output_seq, final_state) = self.lstm1.forward(input, None);
                (
                    {
                        let [seq_len, batch_size, _] = output_seq.dims();
                        let reshaped = output_seq.reshape([seq_len, batch_size, 2, 8usize]);
                        reshaped.swap_dims(1, 2)
                    },
                    final_state.hidden,
                    final_state.cell,
                )
            };
            (Y, Y_h, Y_c)
        }
        ");
    }

    #[test]
    fn test_lstm_forward_reverse() {
        let node = create_lstm_node("lstm1", LstmDirection::Reverse, false, 3);
        let code = codegen_forward_default(&node);
        // Note: reverse is now handled by the LSTM module's config, not by flip() in codegen
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<B, 3>,
            W: Tensor<B, 3>,
            R: Tensor<B, 3>,
            B: Tensor<B, 2>,
        ) -> (Tensor<B, 4>, Tensor<B, 3>, Tensor<B, 3>) {
            let (Y, Y_h, Y_c) = {
                let (output_seq, final_state) = self.lstm1.forward(input, None);
                (
                    output_seq.unsqueeze_dims::<4>(&[1]),
                    final_state.hidden.unsqueeze_dims::<3>(&[0]),
                    final_state.cell.unsqueeze_dims::<3>(&[0]),
                )
            };
            (Y, Y_h, Y_c)
        }
        ");
    }
}
