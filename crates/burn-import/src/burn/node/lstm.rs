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
/// to Burn's individual GateController structure.
///
/// ONNX LSTM weight layout:
/// - W: `[num_directions, 4*hidden_size, input_size]` - gates ordered as [i, o, f, c]
/// - R: `[num_directions, 4*hidden_size, hidden_size]` - gates ordered as [i, o, f, c]
/// - B: `[num_directions, 8*hidden_size]` - Wb[i,o,f,c] then Rb[i,o,f,c]
///
/// Burn LSTM structure (per direction):
/// - input_gate.input_transform: weight `[input_size, hidden_size]`, bias `[hidden_size]`
/// - input_gate.hidden_transform: weight `[hidden_size, hidden_size]`, bias `[hidden_size]`
/// - forget_gate.input_transform/hidden_transform: same structure
/// - output_gate.input_transform/hidden_transform: same structure
/// - cell_gate.input_transform/hidden_transform: same structure
fn collect_lstm_snapshots(
    field_name: &str,
    inputs: &[Argument],
    config: &onnx_ir::lstm::LstmConfig,
) -> Vec<TensorSnapshot> {
    use burn::module::ParamId;
    use burn::tensor::{DType, TensorData};
    use burn_store::TensorSnapshotError;
    use onnx_ir::ir::ArgType;
    use std::rc::Rc;

    let mut snapshots = Vec::new();

    // Get W (input weights), R (hidden weights), and optionally B (biases)
    let w_input = inputs.get(1);
    let r_input = inputs.get(2);
    let b_input = inputs.get(3).filter(|arg| !arg.is_optional());

    // Get dimensions from W tensor
    let (input_size, hidden_size, _num_directions) = match w_input {
        Some(w) => match &w.ty {
            ArgType::Tensor(t) => {
                if let Some(shape) = &t.static_shape {
                    // W shape: [num_directions, 4*hidden_size, input_size]
                    let num_dirs = shape[0];
                    let four_hidden = shape[1];
                    let input_sz = shape[2];
                    (input_sz, four_hidden / 4, num_dirs)
                } else {
                    return vec![];
                }
            }
            _ => return vec![],
        },
        None => return vec![],
    };

    // Get dtype from W tensor
    let dtype = match w_input {
        Some(w) => match &w.ty {
            ArgType::Tensor(t) => t.dtype,
            _ => DType::F32,
        },
        None => DType::F32,
    };

    // ONNX gate order: i(input), o(output), f(forget), c(cell)
    // Burn gate order: input_gate, forget_gate, output_gate, cell_gate
    // Mapping: ONNX[0]->input, ONNX[1]->output, ONNX[2]->forget, ONNX[3]->cell
    let onnx_to_burn_gate_order = [0usize, 2, 1, 3]; // input, forget, output, cell
    let gate_names = ["input_gate", "forget_gate", "output_gate", "cell_gate"];

    // Helper to create container stack for LSTM
    let make_container_stack =
        |container_type: &str| -> Vec<String> { vec![format!("Struct:{}", container_type)] };

    // Clone inputs for use in closures
    let w_input_clone = w_input.cloned();
    let r_input_clone = r_input.cloned();
    let b_input_clone = b_input.cloned();

    // Determine direction prefixes based on LSTM type
    let direction_prefixes: Vec<&str> = match config.direction {
        LstmDirection::Forward | LstmDirection::Reverse => vec![""],
        LstmDirection::Bidirectional => vec!["forward.", "reverse."],
    };

    for (dir_idx, dir_prefix) in direction_prefixes.iter().enumerate() {
        for (gate_idx, gate_name) in gate_names.iter().enumerate() {
            let onnx_gate_idx = onnx_to_burn_gate_order[gate_idx];

            // Input transform weight: slice from W
            // W shape: [num_directions, 4*hidden_size, input_size]
            // We need: [input_size, hidden_size] (transposed)
            {
                let path = format!(
                    "{}.{}{}.input_transform.weight",
                    field_name, dir_prefix, gate_name
                );
                let path_stack: Vec<String> = path.split('.').map(String::from).collect();
                let w_clone = w_input_clone.clone();
                let h = hidden_size;
                let inp = input_size;
                let d_idx = dir_idx;
                let g_idx = onnx_gate_idx;
                let dt = dtype;

                let data_fn = Rc::new(move || -> Result<TensorData, TensorSnapshotError> {
                    let w_data = w_clone.as_ref().and_then(|w| w.value()).ok_or_else(|| {
                        TensorSnapshotError::DataError(
                            "Failed to extract W tensor data".to_string(),
                        )
                    })?;

                    // Extract slice for this direction and gate
                    // W layout: [num_directions, 4*hidden_size, input_size]
                    let start_row = g_idx * h;
                    let end_row = start_row + h;

                    // Create output tensor with transposed shape [input_size, hidden_size]
                    let mut output = vec![0u8; inp * h * element_size(dt)];

                    // Transpose while copying: ONNX [hidden, input] -> Burn [input, hidden]
                    copy_and_transpose_slice(
                        w_data.as_bytes(),
                        &mut output,
                        d_idx,
                        start_row,
                        end_row,
                        inp, // cols in source (input_size)
                        h,   // rows to extract (hidden_size)
                        dt,
                    );

                    Ok(TensorData::from_bytes_vec(output, vec![inp, h], dt))
                });

                snapshots.push(TensorSnapshot::from_closure(
                    data_fn,
                    dtype,
                    vec![input_size, hidden_size],
                    path_stack,
                    make_container_stack("Linear"),
                    ParamId::new(),
                ));
            }

            // Input transform bias: slice from B (first half: Wb)
            if b_input.is_some() {
                let path = format!(
                    "{}.{}{}.input_transform.bias",
                    field_name, dir_prefix, gate_name
                );
                let path_stack: Vec<String> = path.split('.').map(String::from).collect();
                let b_clone = b_input_clone.clone();
                let h = hidden_size;
                let d_idx = dir_idx;
                let g_idx = onnx_gate_idx;
                let dt = dtype;

                let data_fn = Rc::new(move || -> Result<TensorData, TensorSnapshotError> {
                    let b_data = b_clone.as_ref().and_then(|b| b.value()).ok_or_else(|| {
                        TensorSnapshotError::DataError(
                            "Failed to extract B tensor data".to_string(),
                        )
                    })?;

                    // B layout: [num_directions, 8*hidden_size]
                    // First 4*hidden_size are Wb (input biases), next 4*hidden_size are Rb (hidden biases)
                    // For Burn, bias = Wb + Rb for each gate

                    let elem_size = element_size(dt);
                    let eight_h = 8 * h;

                    // Wb offset for this gate
                    let wb_offset = d_idx * eight_h + g_idx * h;
                    // Rb offset for this gate
                    let rb_offset = d_idx * eight_h + 4 * h + g_idx * h;

                    // Sum Wb + Rb element-wise
                    let output = add_bias_slices(
                        b_data.as_bytes(),
                        wb_offset * elem_size,
                        rb_offset * elem_size,
                        h,
                        dt,
                    );

                    Ok(TensorData::from_bytes_vec(output, vec![h], dt))
                });

                snapshots.push(TensorSnapshot::from_closure(
                    data_fn,
                    dtype,
                    vec![hidden_size],
                    path_stack,
                    make_container_stack("Linear"),
                    ParamId::new(),
                ));
            }

            // Hidden transform weight: slice from R
            // R shape: [num_directions, 4*hidden_size, hidden_size]
            // We need: [hidden_size, hidden_size] (transposed)
            {
                let path = format!(
                    "{}.{}{}.hidden_transform.weight",
                    field_name, dir_prefix, gate_name
                );
                let path_stack: Vec<String> = path.split('.').map(String::from).collect();
                let r_clone = r_input_clone.clone();
                let h = hidden_size;
                let d_idx = dir_idx;
                let g_idx = onnx_gate_idx;
                let dt = dtype;

                let data_fn = Rc::new(move || -> Result<TensorData, TensorSnapshotError> {
                    let r_data = r_clone.as_ref().and_then(|r| r.value()).ok_or_else(|| {
                        TensorSnapshotError::DataError(
                            "Failed to extract R tensor data".to_string(),
                        )
                    })?;

                    // Extract slice for this direction and gate
                    // R layout: [num_directions, 4*hidden_size, hidden_size]
                    let start_row = g_idx * h;
                    let end_row = start_row + h;

                    // Create output tensor with transposed shape [hidden_size, hidden_size]
                    let mut output = vec![0u8; h * h * element_size(dt)];

                    // Transpose while copying: ONNX [hidden, hidden] -> Burn [hidden, hidden]
                    copy_and_transpose_slice(
                        r_data.as_bytes(),
                        &mut output,
                        d_idx,
                        start_row,
                        end_row,
                        h, // cols in source (hidden_size)
                        h, // rows to extract (hidden_size)
                        dt,
                    );

                    Ok(TensorData::from_bytes_vec(output, vec![h, h], dt))
                });

                snapshots.push(TensorSnapshot::from_closure(
                    data_fn,
                    dtype,
                    vec![hidden_size, hidden_size],
                    path_stack,
                    make_container_stack("Linear"),
                    ParamId::new(),
                ));
            }

            // Hidden transform bias: slice from B (second half: Rb)
            // Note: In Burn, bias is applied once per gate (in input_transform),
            // but ONNX has separate Wb and Rb. We add Rb to the hidden_transform bias.
            if b_input.is_some() {
                let path = format!(
                    "{}.{}{}.hidden_transform.bias",
                    field_name, dir_prefix, gate_name
                );
                let path_stack: Vec<String> = path.split('.').map(String::from).collect();
                let h = hidden_size;
                let dt = dtype;

                // Burn's GateController actually adds biases from both transforms,
                // so we need to provide zero biases for hidden_transform to avoid double-adding.
                // The full bias (Wb + Rb) is already in input_transform.bias
                let data_fn = Rc::new(move || -> Result<TensorData, TensorSnapshotError> {
                    let elem_size = element_size(dt);
                    let output = vec![0u8; h * elem_size];
                    Ok(TensorData::from_bytes_vec(output, vec![h], dt))
                });

                snapshots.push(TensorSnapshot::from_closure(
                    data_fn,
                    dtype,
                    vec![hidden_size],
                    path_stack,
                    make_container_stack("Linear"),
                    ParamId::new(),
                ));
            }
        }
    }

    snapshots
}

/// Get the size in bytes for a given dtype
fn element_size(dtype: burn::tensor::DType) -> usize {
    use burn::tensor::DType;
    match dtype {
        DType::F32 | DType::I32 | DType::U32 => 4,
        DType::F64 | DType::I64 | DType::U64 => 8,
        DType::F16 | DType::BF16 | DType::I16 => 2,
        DType::I8 | DType::U8 | DType::Bool => 1,
        _ => 4, // default
    }
}

/// Copy and transpose a slice from a 3D packed weight tensor
/// Source layout: [num_directions, rows, cols]
/// Output layout: [cols, extracted_rows] (transposed)
#[allow(clippy::too_many_arguments)]
fn copy_and_transpose_slice(
    src: &[u8],
    dst: &mut [u8],
    dir_idx: usize,
    start_row: usize,
    end_row: usize,
    cols: usize,
    rows_to_extract: usize,
    dtype: burn::tensor::DType,
) {
    use burn::tensor::DType;

    let elem_size = element_size(dtype);
    let total_rows = end_row; // This is based on 4*hidden_size
    let _ = total_rows; // Avoid unused warning

    // Calculate source offset for this direction
    // Source is [num_directions, 4*hidden_size, cols]
    // For direction dir_idx, gate slice starts at start_row
    let dir_stride = 4 * rows_to_extract * cols; // 4*hidden_size * cols per direction
    let src_base = dir_idx * dir_stride * elem_size;

    // Transpose: src[r, c] -> dst[c, r]
    match dtype {
        DType::F32 => {
            transpose_slice_typed::<f32>(src, dst, src_base, start_row, cols, rows_to_extract)
        }
        DType::F64 => {
            transpose_slice_typed::<f64>(src, dst, src_base, start_row, cols, rows_to_extract)
        }
        DType::F16 => {
            transpose_slice_typed::<half::f16>(src, dst, src_base, start_row, cols, rows_to_extract)
        }
        DType::BF16 => transpose_slice_typed::<half::bf16>(
            src,
            dst,
            src_base,
            start_row,
            cols,
            rows_to_extract,
        ),
        DType::I32 => {
            transpose_slice_typed::<i32>(src, dst, src_base, start_row, cols, rows_to_extract)
        }
        DType::I64 => {
            transpose_slice_typed::<i64>(src, dst, src_base, start_row, cols, rows_to_extract)
        }
        _ => panic!("Unsupported dtype for LSTM weight transpose: {:?}", dtype),
    }
}

fn transpose_slice_typed<T: Copy + Default>(
    src: &[u8],
    dst: &mut [u8],
    src_base: usize,
    start_row: usize,
    cols: usize,
    rows_to_extract: usize,
) {
    let elem_size = std::mem::size_of::<T>();

    // Source elements pointer
    let src_elements: &[T] =
        unsafe { std::slice::from_raw_parts(src.as_ptr() as *const T, src.len() / elem_size) };

    // Destination elements pointer
    let dst_elements: &mut [T] = unsafe {
        std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut T, dst.len() / elem_size)
    };

    // Calculate base index in elements (not bytes)
    let src_base_elem = src_base / elem_size;

    // Transpose: src[start_row + r, c] -> dst[c, r]
    for r in 0..rows_to_extract {
        for c in 0..cols {
            let src_idx = src_base_elem + (start_row + r) * cols + c;
            let dst_idx = c * rows_to_extract + r;
            dst_elements[dst_idx] = src_elements[src_idx];
        }
    }
}

/// Add two bias slices element-wise (Wb + Rb)
fn add_bias_slices(
    src: &[u8],
    offset1: usize,
    offset2: usize,
    count: usize,
    dtype: burn::tensor::DType,
) -> Vec<u8> {
    use burn::tensor::DType;

    match dtype {
        DType::F32 => add_bias_typed::<f32>(src, offset1, offset2, count),
        DType::F64 => add_bias_typed::<f64>(src, offset1, offset2, count),
        DType::F16 => add_bias_typed_f16(src, offset1, offset2, count),
        DType::BF16 => add_bias_typed_bf16(src, offset1, offset2, count),
        _ => panic!("Unsupported dtype for LSTM bias addition: {:?}", dtype),
    }
}

fn add_bias_typed<T: Copy + Default + std::ops::Add<Output = T>>(
    src: &[u8],
    offset1: usize,
    offset2: usize,
    count: usize,
) -> Vec<u8> {
    let elem_size = std::mem::size_of::<T>();

    let src_elements: &[T] =
        unsafe { std::slice::from_raw_parts(src.as_ptr() as *const T, src.len() / elem_size) };

    let idx1 = offset1 / elem_size;
    let idx2 = offset2 / elem_size;

    let mut result = vec![T::default(); count];
    for i in 0..count {
        result[i] = src_elements[idx1 + i] + src_elements[idx2 + i];
    }

    // Convert back to bytes
    let result_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(result.as_ptr() as *const u8, count * elem_size) };
    result_bytes.to_vec()
}

fn add_bias_typed_f16(src: &[u8], offset1: usize, offset2: usize, count: usize) -> Vec<u8> {
    let elem_size = std::mem::size_of::<half::f16>();

    let src_elements: &[half::f16] = unsafe {
        std::slice::from_raw_parts(src.as_ptr() as *const half::f16, src.len() / elem_size)
    };

    let idx1 = offset1 / elem_size;
    let idx2 = offset2 / elem_size;

    let mut result = vec![half::f16::ZERO; count];
    for i in 0..count {
        result[i] = src_elements[idx1 + i] + src_elements[idx2 + i];
    }

    // Convert back to bytes
    let result_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(result.as_ptr() as *const u8, count * elem_size) };
    result_bytes.to_vec()
}

fn add_bias_typed_bf16(src: &[u8], offset1: usize, offset2: usize, count: usize) -> Vec<u8> {
    let elem_size = std::mem::size_of::<half::bf16>();

    let src_elements: &[half::bf16] = unsafe {
        std::slice::from_raw_parts(src.as_ptr() as *const half::bf16, src.len() / elem_size)
    };

    let idx1 = offset1 / elem_size;
    let idx2 = offset2 / elem_size;

    let mut result = vec![half::bf16::ZERO; count];
    for i in 0..count {
        result[i] = src_elements[idx1 + i] + src_elements[idx2 + i];
    }

    // Convert back to bytes
    let result_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(result.as_ptr() as *const u8, count * elem_size) };
    result_bytes.to_vec()
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
