use burn_core as burn;
use burn_core::config::Config;

use alloc::vec::Vec;
use burn::tensor::ops::IntElem;

use burn::tensor::{Bool, ElementConversion, Int, Shape, Tensor, TensorData, backend::Backend};

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

/// Generate a padding attention mask.
pub struct GeneratePaddingMask<B: Backend> {
    /// The generated tensor.
    pub tensor: Tensor<B, 2, Int>,

    /// The generated mask.
    pub mask: Tensor<B, 2, Bool>,
}

/// Defines an enumeration to specify sequence length options for padding
#[derive(Config, Debug, Copy)]
pub enum SeqLengthOption {
    /// No maximum length; use the longest sequence
    NoMax,
    /// Maximum length specified, truncate if necessary
    Max(usize),
    /// Fixed length, pad or truncate to this exact length
    Fixed(usize),
}

impl From<Option<usize>> for SeqLengthOption {
    fn from(val: Option<usize>) -> Self {
        match val {
            Some(max) => SeqLengthOption::Max(max),
            None => SeqLengthOption::NoMax,
        }
    }
}

/// Generates a padding attention mask for a batch of token sequences.
///
/// # Arguments
///
/// * `pad_token` - The token ID used for padding
/// * `tokens_list` - Vector of token sequences (each sequence is a vector of token IDs)
/// * `seq_length` - Sequence length option (NoMax, Max, or Fixed)
/// * `device` - The device for tensor operations
///
/// # Returns
///
/// A `GeneratePaddingMask` containing the padded tensor and corresponding mask
pub fn generate_padding_mask<B: Backend>(
    pad_token: usize,
    tokens_list: Vec<Vec<usize>>,
    seq_length: impl Into<SeqLengthOption>,
    device: &B::Device,
) -> GeneratePaddingMask<B> {
    let tokens_max = || {
        tokens_list
            .iter()
            .map(|tokens| tokens.len())
            .max()
            .unwrap_or(1)
    };

    let size = match seq_length.into() {
        SeqLengthOption::NoMax => tokens_max(),
        SeqLengthOption::Max(max) => usize::min(tokens_max(), max),
        SeqLengthOption::Fixed(limit) => limit,
    };
    let batch_size = tokens_list.len();

    let mut tensor = Tensor::zeros([batch_size, size], device);
    tensor = tensor.add_scalar(pad_token as i64);

    for (index, tokens) in tokens_list.into_iter().enumerate() {
        let seq_length = tokens.len().min(size);
        tensor = tensor.slice_assign(
            [index..index + 1, 0..seq_length],
            Tensor::from_data(
                TensorData::new(
                    tokens
                        .into_iter()
                        .take(size)
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
    use alloc::vec;
    use burn::tensor::TensorData;

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
