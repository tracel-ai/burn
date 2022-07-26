use crate::{
    backend::ndarray::{BatchMatrix, NdArrayTensor},
    TensorOpsMatmul,
};
use ndarray::{Dim, Dimension, LinalgScalar};

impl<P, const D: usize> TensorOpsMatmul<P, D> for NdArrayTensor<P, D>
where
    P: Clone + LinalgScalar + Default + std::fmt::Debug,
    Dim<[usize; D]>: Dimension,
{
    fn matmul(&self, other: &Self) -> Self {
        let batch_self = BatchMatrix::from_ndarray(self.array.clone(), self.shape.clone());
        let batch_other = BatchMatrix::from_ndarray(other.array.clone(), other.shape.clone());

        let self_iter = batch_self.arrays.iter();
        let other_iter = batch_other.arrays.iter();
        let arrays = self_iter
            .zip(other_iter)
            .map(|(lhs, rhs)| lhs.dot(rhs))
            .map(|output| output.into_shared())
            .collect();

        let mut shape = self.shape.clone();
        shape.dims[D - 1] = other.shape.dims[D - 1];
        let output = BatchMatrix::new(arrays, shape.clone());

        Self::from_bmatrix(output)
    }
}

#[cfg(test)]
mod tests {
    use crate::{backend::ndarray::NdArrayTensor, Data, TensorBase, TensorOpsMatmul};

    #[test]
    fn should_matmul_d2() {
        let data_1: Data<f64, 2> = Data::from([[1.0, 7.0], [2.0, 3.0], [1.0, 5.0]]);
        let data_2: Data<f64, 2> = Data::from([[4.0, 7.0, 5.0], [2.0, 3.0, 5.0]]);

        let tensor_1 = NdArrayTensor::from_data(data_1.clone());
        let tensor_2 = NdArrayTensor::from_data(data_2.clone());

        let tensor_3 = tensor_1.matmul(&tensor_2);

        assert_eq!(
            tensor_3.into_data(),
            Data::from([[18.0, 28.0, 40.0], [14.0, 23.0, 25.0], [14.0, 22.0, 30.0]])
        );
    }

    #[test]
    fn should_matmul_d3() {
        let data_1: Data<f64, 3> = Data::from([[[1.0, 7.0], [2.0, 3.0]]]);
        let data_2: Data<f64, 3> = Data::from([[[4.0, 7.0], [2.0, 3.0]]]);

        let tensor_1 = NdArrayTensor::from_data(data_1.clone());
        let tensor_2 = NdArrayTensor::from_data(data_2.clone());

        let tensor_3 = tensor_1.matmul(&tensor_2);

        assert_eq!(
            tensor_3.into_data(),
            Data::from([[[18.0, 28.0], [14.0, 23.0]]])
        );
    }
}
