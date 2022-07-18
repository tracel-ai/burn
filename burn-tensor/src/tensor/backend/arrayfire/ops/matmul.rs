use crate::{backend::arrayfire::ArrayfireTensor, Shape, TensorOpsMatmul};
use arrayfire::{FloatingPoint, HasAfEnum};

impl<P: HasAfEnum + FloatingPoint, const D: usize> TensorOpsMatmul<P, D> for ArrayfireTensor<P, D> {
    fn matmul(&self, other: &Self) -> Self {
        self.set_backend_binary_ops(other);

        let array = arrayfire::matmul(
            &self.array,
            &other.array,
            arrayfire::MatProp::NONE,
            arrayfire::MatProp::NONE,
        );
        let device = self.device;
        let shape = Shape::from(array.dims());

        Self {
            array,
            shape,
            device,
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{backend::arrayfire::Device, Data, TensorBase};

    #[test]
    fn should_support_matmul_2_dims() {
        let data_1 = Data::from([[4.0, 3.0], [8.0, 7.0]]);
        let data_2 = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor_1 = ArrayfireTensor::<f64, 2>::from_data(data_1, Device::CPU);
        let tensor_2 = ArrayfireTensor::<f64, 2>::from_data(data_2, Device::CPU);

        let data_actual = tensor_1.matmul(&tensor_2).into_data();

        let data_expected = Data::from([[9.0, 16.0, 23.0], [21.0, 36.0, 51.0]]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    #[ignore = "batch operation not supported yet..."]
    fn should_support_matmul_3_dims() {
        let data_1 = Data::from([[[4.0, 3.0], [8.0, 7.0]], [[4.0, 3.0], [8.0, 7.0]]]);
        let data_2 = Data::from([
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
        ]);
        let tensor_1 = ArrayfireTensor::<f32, 3>::from_data(data_1, Device::CPU);
        let tensor_2 = ArrayfireTensor::<f32, 3>::from_data(data_2, Device::CPU);

        let data_actual = tensor_1.matmul(&tensor_2).into_data();

        let data_expected = Data::from([
            [[9.0, 16.0, 23.0], [21.0, 36.0, 51.0]],
            [[9.0, 16.0, 23.0], [21.0, 36.0, 51.0]],
        ]);
        assert_eq!(data_expected, data_actual);
    }
}
