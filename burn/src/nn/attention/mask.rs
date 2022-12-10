use burn_tensor::{backend::Backend, BoolTensor, ElementConversion, Tensor};

/// Generate an autoregressive attention mask.
///
/// The mask can be used in Transformer modules to train models to generate tensors sequentially.
pub fn generate_autoregressive_mask<B: Backend>(
    batch_size: usize,
    seq_length: usize,
    device: B::Device,
) -> BoolTensor<B, 3> {
    let mut mask = Tensor::<B::IntegerBackend, 3>::zeros([1, seq_length, seq_length]);

    for i in 0..seq_length {
        let values = Tensor::<B::IntegerBackend, 3>::ones([1, 1, seq_length - (i + 1)]);
        mask = mask.index_assign([0..1, i..i + 1, i + 1..seq_length], &values);
    }

    mask = mask.to_device(device).repeat(0, batch_size);

    BoolTensor::from_int_backend(mask.equal_scalar(1_i64.to_elem::<i64>()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn_tensor::Data;

    #[test]
    fn test_generate_autoregressive_mask() {
        let device = <TestBackend as Backend>::Device::default();
        let mask = generate_autoregressive_mask::<TestBackend>(2, 3, device);

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
}
