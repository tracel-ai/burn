use crate::{conv::conv2d_naive, element::NdArrayElement, tensor::NdArrayTensor, NdArrayBackend};
use burn_tensor::{ops::*, Shape};
use std::ops::Add;

impl<E: NdArrayElement> ModuleOps<NdArrayBackend<E>> for NdArrayBackend<E> {
    fn embedding(
        weights: &NdArrayTensor<E, 2>,
        indexes: &NdArrayTensor<i64, 2>,
    ) -> NdArrayTensor<E, 3> {
        let [batch_size, seq_length] = indexes.shape.dims;
        let [_n_embedding, d_model] = weights.shape.dims;

        let mut tensors = Vec::with_capacity(batch_size * seq_length);

        for index in NdArrayBackend::reshape(indexes, Shape::new([batch_size * seq_length]))
            .array
            .iter()
        {
            let index = *index as usize;
            tensors.push(NdArrayBackend::index(
                weights,
                [index..index + 1, 0..d_model],
            ));
        }
        let embedding = NdArrayBackend::cat(&tensors, 0);
        NdArrayBackend::reshape(&embedding, Shape::new([batch_size, seq_length, d_model]))
    }

    fn embedding_backward(
        weights: &NdArrayTensor<E, 2>,
        output: &NdArrayTensor<E, 3>,
        indexes: &NdArrayTensor<i64, 2>,
    ) -> NdArrayTensor<E, 2> {
        let [batch_size, seq_length] = indexes.shape.dims;
        let [_n_embedding, d_model] = weights.shape.dims;

        let mut weights_grad = weights.zeros();
        let output =
            NdArrayBackend::reshape(output, Shape::new([batch_size * seq_length, d_model]));

        for (index_output, index) in
            NdArrayBackend::reshape(indexes, Shape::new([batch_size * seq_length]))
                .array
                .iter()
                .enumerate()
        {
            let index = *index as usize;

            let weights_grad_current =
                NdArrayBackend::index(&weights_grad, [index..index + 1, 0..d_model]);
            let output_grad =
                NdArrayBackend::index(&output, [index_output..index_output + 1, 0..d_model]);

            weights_grad = NdArrayBackend::index_assign(
                &weights_grad,
                [index..index + 1, 0..d_model],
                &output_grad.add(weights_grad_current),
            );
        }

        weights_grad
    }

    fn conv2d(
        x: &NdArrayTensor<E, 4>,
        weight: &NdArrayTensor<E, 4>,
        bias: Option<&NdArrayTensor<E, 1>>,
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> NdArrayTensor<E, 4> {
        let [batch_size, channels_in, heigth, width] = x.shape.dims;
        let mut results = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let x = NdArrayBackend::index(x, [b..b + 1, 0..channels_in, 0..heigth, 0..width]);
            let x = NdArrayBackend::reshape(&x, Shape::new([channels_in, heigth, width]));

            results.push(conv2d_naive(&x, weight, bias, stride, padding));
        }

        NdArrayBackend::cat(&results, 0)
    }
}
