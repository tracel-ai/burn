use crate::{backend::arrayfire::ArrayfireTensor, TensorOpsAdd};
use arrayfire::{ConstGenerator, HasAfEnum};
use std::ops::Add;

impl<P: HasAfEnum, const D: usize> TensorOpsAdd<P, D> for ArrayfireTensor<P, D>
where
    P: ConstGenerator<OutType = P> + Clone + Copy,
{
    fn add(&self, other: &Self) -> Self {
        self.set_backend_binary_ops(other);

        let array = (&self.array).add(&other.array);
        let shape = self.shape.clone();
        let device = self.device;

        Self {
            array,
            shape,
            device,
        }
    }
    fn add_scalar(&self, other: &P) -> Self {
        self.set_backend_single_ops();

        let array = arrayfire::add(&self.array, other, false);
        let shape = self.shape.clone();
        let device = self.device;

        Self {
            array,
            shape,
            device,
        }
    }
}

impl<P: HasAfEnum, const D: usize> std::ops::Add<Self> for ArrayfireTensor<P, D>
where
    P: ConstGenerator<OutType = P> + Clone + Copy,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        TensorOpsAdd::add(&self, &rhs)
    }
}

impl<P: HasAfEnum, const D: usize> std::ops::Add<P> for ArrayfireTensor<P, D>
where
    P: ConstGenerator<OutType = P> + Clone + Copy,
{
    type Output = Self;

    fn add(self, rhs: P) -> Self::Output {
        TensorOpsAdd::add_scalar(&self, &rhs)
    }
}

#[cfg(test)]
mod tests {
    use crate::{backend::arrayfire::Device, Data, TensorBase};

    use super::*;

    #[test]
    fn should_support_add_ops() {
        let data_1 = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let data_2 = Data::from([[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]);
        let data_expected = Data::from([[6.0, 8.0, 10.0], [12.0, 14.0, 16.0]]);
        let tensor_1 = ArrayfireTensor::<f64, 2>::from_data(data_1, Device::CPU);
        let tensor_2 = ArrayfireTensor::<f64, 2>::from_data(data_2, Device::CPU);

        let data_actual = (tensor_1 + tensor_2).into_data();

        assert_eq!(data_expected, data_actual);
    }
}
