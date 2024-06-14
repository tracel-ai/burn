use alloc::vec::Vec;

use crate::tensor::{backend::Backend, Bool, Data, ElementConversion, Int, Shape, Tensor};

/// Generate an autoregressive attention mask.
///
/// The mask can be used in Transformer modules to train models to generate tensors sequentially.
pub fn generate_autoregressive_mask<B: Backend>(
    batch_size: usize,
    seq_length: usize,
    device: &B::Device,
) -> Tensor<B, 3, Bool> {
    // TODO replace with more efficient op of `triu_mask` and `expand`
    let mut mask = Tensor::<B, 3, Int>::zeros([1, seq_length, seq_length], device);

    for i in 0..(seq_length - 1) {
        let values = Tensor::<B, 3, Int>::ones([1, 1, seq_length - (i + 1)], device);
        mask = mask.slice_assign([0..1, i..i + 1, i + 1..seq_length], values);
    }

    mask = mask.repeat(0, batch_size);

    mask.equal_elem(1_i64.elem::<i64>())
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

        if let Some(max_seq_length) = max_seq_length {
            if tokens.len() >= max_seq_length {
                max_size = max_seq_length;
                break;
            }
        }
    }

    let mut tensor = Tensor::zeros([batch_size, max_size], device);
    tensor = tensor.add_scalar(pad_token as i64);

    for (index, tokens) in tokens_list.into_iter().enumerate() {
        let mut seq_length = tokens.len();
        let mut tokens = tokens;

        if let Some(max_seq_length) = max_seq_length {
            if seq_length > max_seq_length {
                seq_length = max_seq_length;
                let _ = tokens.split_off(seq_length);
            }
        }

        tensor = tensor.slice_assign(
            [index..index + 1, 0..tokens.len()],
            Tensor::from_data(
                Data::new(
                    tokens.into_iter().map(|e| (e as i64).elem()).collect(),
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
    use crate::tensor::Data;
    use crate::TestBackend;
    use alloc::vec;

    #[test]
    fn test_generate_autoregressive_mask() {
        let device = <TestBackend as Backend>::Device::default();

        let mask = generate_autoregressive_mask::<TestBackend>(2, 3, &device);

        assert_eq!(
            mask.into_data(),
            Data::from([
                [
                    [false, true, true],
                    [false, false, true],
                    [false, false, false],
                ],
                [
                    [false, true, true],
                    [false, false, true],
                    [false, false, false],
                ]
            ])
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

        assert_eq!(
            mask.mask.into_data(),
            Data::from([
                [false, false, false, true, true, true],
                [false, false, false, true, true, true],
                [false, false, false, false, true, true],
                [false, false, false, false, false, false],
            ])
        );
    }
}
