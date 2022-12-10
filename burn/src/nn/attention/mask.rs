use burn_tensor::{backend::Backend, BoolTensor, ElementConversion, Tensor};

pub fn generate_autoregressive_mask<B: Backend>(
    batch_size: usize,
    seq_length: usize,
    device: B::Device,
) -> BoolTensor<B, 3> {
    let mut mask =
        Tensor::<B::IntegerBackend, 3>::zeros_device([batch_size, seq_length, seq_length], device);

    for i in 0..seq_length {
        let values = Tensor::<B::IntegerBackend, 3>::ones_device(
            [batch_size, 1, seq_length - (i + 1)],
            device,
        );
        mask = mask.index_assign([0..batch_size, i..i + 1, i + 1..seq_length], &values);
    }

    BoolTensor::from_int_backend(mask.equal_scalar(1_i64.to_elem::<i64>()))
}

#[cfg(test)]
mod tests {
    use burn_tensor::Data;

    use super::*;
    use crate::TestBackend;

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
