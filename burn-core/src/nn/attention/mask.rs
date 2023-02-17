use burn_tensor::{backend::Backend, BoolTensor, Data, ElementConversion, Shape, Tensor};

/// Generate an autoregressive attention mask.
///
/// The mask can be used in Transformer modules to train models to generate tensors sequentially.
pub fn generate_autoregressive_mask<B: Backend>(
    batch_size: usize,
    seq_length: usize,
    device: &B::Device,
) -> BoolTensor<B, 3> {
    let mut mask = Tensor::<B::IntegerBackend, 3>::zeros([1, seq_length, seq_length]);

    for i in 0..seq_length {
        let values = Tensor::<B::IntegerBackend, 3>::ones([1, 1, seq_length - (i + 1)]);
        mask = mask.index_assign([0..1, i..i + 1, i + 1..seq_length], values);
    }

    mask = mask.to_device(device).repeat(0, batch_size);

    BoolTensor::from_int_backend(mask.equal_scalar(1_i64.to_elem::<i64>()))
}

pub struct GeneratePaddingMask<B: Backend> {
    pub tensor: Tensor<B::IntegerBackend, 2>,
    pub mask: BoolTensor<B, 2>,
}

/// Generation padding attention mask.
pub fn generate_padding_mask<B: Backend>(
    pad_token: usize,
    tokens_list: Vec<Vec<usize>>,
    max_seq_lenght: Option<usize>,
    device: &B::Device,
) -> GeneratePaddingMask<B> {
    let mut max_size = 0;
    let batch_size = tokens_list.len();

    for tokens in tokens_list.iter() {
        if tokens.len() > max_size {
            max_size = tokens.len();
        }

        if let Some(max_seq_lenght) = max_seq_lenght {
            if tokens.len() >= max_seq_lenght {
                max_size = max_seq_lenght;
                break;
            }
        }
    }

    let mut tensor = Tensor::zeros([batch_size, max_size]);
    tensor = tensor.add_scalar(pad_token as i64);

    for (index, tokens) in tokens_list.into_iter().enumerate() {
        let mut seq_length = tokens.len();
        let mut tokens = tokens;

        if let Some(max_seq_lenght) = max_seq_lenght {
            if seq_length > max_seq_lenght {
                seq_length = max_seq_lenght;
                let _ = tokens.split_off(seq_length);
            }
        }

        tensor = tensor.index_assign(
            [index..index + 1, 0..tokens.len()],
            Tensor::from_data(Data::new(
                tokens.into_iter().map(|e| e as i64).collect(),
                Shape::new([1, seq_length]),
            )),
        );
    }

    let mask = BoolTensor::from_int_backend(tensor.clone().equal_scalar(pad_token as i64))
        .to_device(device);
    let tensor = tensor.to_device(device);

    GeneratePaddingMask { tensor, mask }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn_tensor::Data;

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
