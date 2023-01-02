use crate::{element::TchElement, TchBackend, TchTensor};
use burn_tensor::{ops::ModuleOps, Shape};

impl<E: TchElement> ModuleOps<TchBackend<E>> for TchBackend<E> {
    fn embedding(weights: &TchTensor<E, 2>, indexes: &TchTensor<i64, 2>) -> TchTensor<E, 3> {
        let tensor = tch::Tensor::embedding(&weights.tensor, &indexes.tensor, -1, false, false);
        let shape = Shape::from(tensor.size());

        TchTensor {
            kind: weights.kind,
            tensor,
            shape,
        }
    }

    fn embedding_backward(
        weights: &TchTensor<E, 2>,
        output: &TchTensor<E, 3>,
        indexes: &TchTensor<i64, 2>,
    ) -> TchTensor<E, 2> {
        let [n_embedding, _d_model] = weights.shape.dims;
        let tensor = tch::Tensor::embedding_backward(
            &output.tensor,
            &indexes.tensor,
            n_embedding as i64,
            -1,
            false,
            false,
        );
        let shape = Shape::from(tensor.size());

        TchTensor {
            kind: weights.kind,
            tensor,
            shape,
        }
    }

    fn conv1d(
        x: &TchTensor<E, 3>,
        weight: &TchTensor<E, 3>,
        bias: &Option<TchTensor<E, 1>>,
        stride: usize,
        padding: usize,
        dilatation: usize,
        groups: usize,
    ) -> TchTensor<E, 3> {
        let tensor = tch::Tensor::conv1d(
            &x.tensor,
            &weight.tensor,
            bias.clone().map(|t| t.tensor),
            &[stride as i64],
            &[padding as i64],
            &[dilatation as i64],
            groups as i64,
        );

        let shape = Shape::from(tensor.size());

        TchTensor {
            kind: weight.kind,
            tensor,
            shape,
        }
    }
}
