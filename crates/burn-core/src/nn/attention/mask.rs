use alloc::vec::Vec;
use burn_tensor::ops::IntElem;

use crate::tensor::{Bool, ElementConversion, Int, Shape, Tensor, TensorData, backend::Backend};

/// Generate an autoregressive attention mask.
///
/// The mask can be used in Transformer modules to train models to generate tensors sequentially.
pub fn generate_autoregressive_mask<B: Backend>(
    batch_size: usize,
    seq_length: usize,
    device: &B::Device,
) -> Tensor<B, 3, Bool> {
    let mask = Tensor::<B, 2, Bool>::tril_mask([seq_length, seq_length], 0, device);
    mask.expand([batch_size, seq_length, seq_length])
}

/// Generate a windowed causal attention mask with optional sink tokens.
///
/// - Allows attending to at most `sink_tokens` tokens at the start, plus the
///   last `window_len` tokens before the current position (inclusive).
/// - When `window_len` is `None`, this reduces to a full causal mask.
pub fn generate_windowed_causal_mask<B: Backend>(
    batch_size: usize,
    seq_length: usize,
    window_len: Option<usize>,
    sink_tokens: usize,
    device: &B::Device,
) -> Tensor<B, 3, Bool> {
    // Base full-causal mask for future positions (True = masked).
    let base = Tensor::<B, 2, Bool>::tril_mask([seq_length, seq_length], 0, device);

    if let Some(w) = window_len {
        // Build per-row mask to zero out keys older than i - w, except sink region.
        let mut mask = Tensor::<B, 3, Bool>::empty([1, seq_length, seq_length], device);
        for i in 0..seq_length {
            // Positions allowed: [0..sink_tokens) U [max(0, i-w) .. i]
            let start = i.saturating_sub(w);
            let mut row = Tensor::<B, 1, Bool>::full([seq_length], true, device);
            if sink_tokens > 0 {
                row = row.slice_assign(0..sink_tokens, Tensor::full([sink_tokens], false, device));
            }
            let to_i = Tensor::<B, 1, Bool>::full([i + 1], false, device); // unmask [0..i]
            row = row.slice_assign(0..i + 1, to_i);
            if start > 0 {
                let rem = Tensor::<B, 1, Bool>::full([start], true, device); // re-mask [0..start)
                row = row.slice_assign(0..start, rem);
            }
            // `row` already masks the future and out-of-window positions.
            mask = mask.slice_assign(
                [0..1, i..i + 1, 0..seq_length],
                row.reshape([1, 1, seq_length]),
            );
        }
        mask.expand([batch_size, seq_length, seq_length])
    } else {
        base.expand([batch_size, seq_length, seq_length])
    }
}

/// Generate a padding attention mask.
pub struct GeneratePaddingMask<B: Backend> {
    /// The generated tensor.
    pub tensor: Tensor<B, 2, Int>,

    /// The generated mask.
    pub mask: Tensor<B, 2, Bool>,
}

/// Generation padding attention mask.
pub fn generate_padding_mask<B: Backend>(
    pad_token: usize,
    tokens_list: Vec<Vec<usize>>,
    max_seq_length: Option<usize>,
    device: &B::Device,
) -> GeneratePaddingMask<B> {
    let mut max_size = 0;
    let batch_size = tokens_list.len();

    for tokens in tokens_list.iter() {
        if tokens.len() > max_size {
            max_size = tokens.len();
        }

        if let Some(max_seq_length) = max_seq_length
            && tokens.len() >= max_seq_length
        {
            max_size = max_seq_length;
            break;
        }
    }

    let mut tensor = Tensor::zeros([batch_size, max_size], device);
    tensor = tensor.add_scalar(pad_token as i64);

    for (index, tokens) in tokens_list.into_iter().enumerate() {
        let seq_length = tokens.len().min(max_size);
        tensor = tensor.slice_assign(
            [index..index + 1, 0..seq_length],
            Tensor::from_data(
                TensorData::new(
                    tokens
                        .into_iter()
                        .take(max_size)
                        .map(|e| (e as i64).elem::<IntElem<B>>())
                        .collect(),
                    Shape::new([1, seq_length]),
                ),
                device,
            ),
        );
    }

    let mask = tensor.clone().equal_elem(pad_token as i64);

    GeneratePaddingMask { tensor, mask }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use crate::tensor::TensorData;
    use alloc::vec;

    #[test]
    fn test_generate_autoregressive_mask() {
        let device = <TestBackend as Backend>::Device::default();

        let mask = generate_autoregressive_mask::<TestBackend>(2, 3, &device);

        mask.into_data().assert_eq(
            &TensorData::from([
                [
                    [false, true, true],
                    [false, false, true],
                    [false, false, false],
                ],
                [
                    [false, true, true],
                    [false, false, true],
                    [false, false, false],
                ],
            ]),
            false,
        );
    }

    #[test]
    fn test_generate_padding_mask() {
        let device = <TestBackend as Backend>::Device::default();
        let tokens = vec![
            vec![3, 3, 3],
            vec![3, 3, 3],
            vec![3, 3, 3, 4],
            vec![3, 3, 3, 4, 10, 15],
        ];

        let mask = generate_padding_mask::<TestBackend>(0, tokens, None, &device);

        mask.mask.into_data().assert_eq(
            &TensorData::from([
                [false, false, false, true, true, true],
                [false, false, false, true, true, true],
                [false, false, false, false, true, true],
                [false, false, false, false, false, false],
            ]),
            false,
        );
    }

    // Additional windowed causal mask tests are in integration tests under
    // crates/burn-core/tests/attention_mask.rs
}
