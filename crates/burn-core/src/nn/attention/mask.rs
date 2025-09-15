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

/// Generate a 1D padding mask from sequence lengths.
///
/// The resulting mask has shape `[batch_size, max_len]` with `true` marking padding positions.
pub fn lengths_to_mask<B: Backend>(
    lengths: &[usize],
    max_len: usize,
    device: &B::Device,
) -> Tensor<B, 2, Bool> {
    let batch = lengths.len();
    // Start with all masked (true), unmask valid tokens per row.
    let mut mask = Tensor::<B, 2, Bool>::full([batch, max_len], true, device);
    for (i, &len_i) in lengths.iter().enumerate() {
        let keep = len_i.min(max_len);
        if keep > 0 {
            let unmask = Tensor::<B, 2, Bool>::full([1, keep], false, device);
            mask = mask.slice_assign([i..i + 1, 0..keep], unmask);
        }
    }
    mask
}

/// Generate a causal mask for a single sequence length (no batch), shape `[seq_len, seq_len]`.
///
/// True indicates a masked position (future positions are masked).
pub fn generate_causal_mask_1d<B: Backend>(seq_len: usize, device: &B::Device) -> Tensor<B, 2, Bool> {
    Tensor::<B, 2, Bool>::tril_mask([seq_len, seq_len], 0, device)
}

/// Generate a chunked causal mask for streaming encoders.
///
/// Returns a boolean mask of shape `[seq_len, seq_len]` where `true` marks masked (disallowed)
/// positions and `false` marks allowed positions. Each position `i` can attend to indices in
/// `[start .. end)` where `start = max(0, (i / chunk_size - num_left_chunks)*chunk_size)` and
/// `end = min(((i / chunk_size) + 1)*chunk_size, seq_len)`. Future positions remain masked.
pub fn generate_chunked_causal_mask_1d<B: Backend>(
    seq_len: usize,
    chunk_size: usize,
    num_left_chunks: isize,
    device: &B::Device,
) -> Tensor<B, 2, Bool> {
    // Start fully masked.
    let mut mask = Tensor::<B, 2, Bool>::full([seq_len, seq_len], true, device);
    if chunk_size == 0 {
        return mask;
    }
    for i in 0..seq_len {
        let chunk_idx = i / chunk_size;
        let start = if num_left_chunks < 0 {
            0usize
        } else {
            let left = (chunk_idx as isize - num_left_chunks).max(0) as usize;
            left * chunk_size
        };
        let mut end = ((chunk_idx + 1) * chunk_size).min(seq_len);
        // Enforce causality (no future positions): end at i+1
        if end > i + 1 {
            end = i + 1;
        }
        if end > start {
            let unmask = Tensor::<B, 2, Bool>::full([1, end - start], false, device);
            mask = mask.slice_assign([i..i + 1, start..end], unmask);
        }
    }
    mask
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
}
