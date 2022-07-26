use crate::{backend::ndarray::NdArrayTensor, TensorOpsMatmul};
use ndarray::LinalgScalar;

impl<P, const D: usize> TensorOpsMatmul<f32, D> for NdArrayTensor<P, D>
where
    P: Clone + LinalgScalar + Default,
{
    fn matmul(&self, other: &Self) -> Self {
        let self_iter = self.arrays.iter();
        let other_iter = other.arrays.iter();
        let arrays = self_iter
            .zip(other_iter)
            .map(|(lhs, rhs)| lhs.dot(rhs))
            .map(|output| output.into_shared())
            .collect();

        let mut shape = self.shape.clone();
        shape.dims[D - 1] = other.shape.dims[D - 1];

        Self { arrays, shape }
    }
}

#[cfg(test)]
mod tests {
    use crate::{backend::ndarray::NdArrayTensor, Data, Shape, TensorBase, TensorOpsMatmul};

    #[test]
    fn should_matmul_d2() {
        let data_1: Data<f64, 2> = Data::from([[1.0, 7.0], [2.0, 3.0], [1.0, 5.0]]);
        let data_2: Data<f64, 2> = Data::from([[4.0, 7.0, 5.0], [2.0, 3.0, 5.0]]);

        let tensor_1 = NdArrayTensor::from_data(data_1.clone());
        let tensor_2 = NdArrayTensor::from_data(data_2.clone());

        let tensor_3 = tensor_1.matmul(&tensor_2);

        assert_eq!(tensor_3.shape, Shape::new([3, 3]));
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
