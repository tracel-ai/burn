use crate::{backend::arrayfire::ArrayfireTensor, TensorOpsMul};
use arrayfire::{ConstGenerator, HasAfEnum};

impl<P: HasAfEnum, const D: usize> TensorOpsMul<P, D> for ArrayfireTensor<P, D>
where
    P: ConstGenerator<OutType = P> + Clone + Copy,
{
    fn mul(&self, other: &Self) -> Self {
        self.set_backend_binary_ops(other);
        let array = arrayfire::mul(&self.array, &other.array, false);
        let shape = self.shape.clone();
        let device = self.device;

        Self {
            array,
            shape,
            device,
        }
    }
    fn mul_scalar(&self, other: &P) -> Self {
        self.set_backend_single_ops();
        let array = arrayfire::mul(&self.array, other, false);
        let shape = self.shape.clone();
        let device = self.device;

        Self {
            array,
            shape,
            device,
        }
    }
}

impl<P: HasAfEnum, const D: usize> std::ops::Mul<P> for ArrayfireTensor<P, D>
where
    P: ConstGenerator<OutType = P> + Clone + Copy,
{
    type Output = ArrayfireTensor<P, D>;

    fn mul(self, rhs: P) -> Self::Output {
        TensorOpsMul::mul_scalar(&self, &rhs)
    }
}

impl<P: HasAfEnum, const D: usize> std::ops::Mul<ArrayfireTensor<P, D>> for ArrayfireTensor<P, D>
where
    P: ConstGenerator<OutType = P> + Clone + Copy,
{
    type Output = ArrayfireTensor<P, D>;

    fn mul(self, rhs: Self) -> Self::Output {
        TensorOpsMul::mul(&self, &rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{backend::arrayfire::Device, Data, TensorBase};

    #[test]
    fn should_support_mul_ops() {
        let data_1 = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let data_2 = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor_1 = ArrayfireTensor::<f64, 2>::from_data(data_1, Device::CPU);
        let tensor_2 = ArrayfireTensor::<f64, 2>::from_data(data_2, Device::CPU);

        let data_actual = tensor_1.mul(&tensor_2).into_data();

        let data_expected = Data::from([[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_mul_scalar_ops() {
        let data = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let scalar = 2.0;
        let tensor = ArrayfireTensor::<f64, 2>::from_data(data, Device::CPU);

        let data_actual = tensor.mul_scalar(&scalar).into_data();

        let data_expected = Data::from([[0.0, 2.0, 4.0], [6.0, 8.0, 10.0]]);
        assert_eq!(data_expected, data_actual);
    }
}
