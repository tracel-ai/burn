use burn_core as burn;

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
