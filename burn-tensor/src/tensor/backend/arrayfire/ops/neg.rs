use arrayfire::{ConstGenerator, HasAfEnum};
use num_traits::One;
use std::ops::Neg;

use crate::{backend::arrayfire::ArrayfireTensor, TensorOpsMul, TensorOpsNeg};

impl<P: HasAfEnum, const D: usize> TensorOpsNeg<P, D> for ArrayfireTensor<P, D>
where
    P: ConstGenerator<OutType = P> + Neg<Output = P> + One + Neg + Clone + Copy,
{
    fn neg(&self) -> Self {
        self.set_backend_single_ops();
        let minus_one = Neg::neg(P::one());
        self.mul_scalar(&minus_one)
    }
}

impl<P: HasAfEnum, const D: usize> std::ops::Neg for ArrayfireTensor<P, D>
where
    P: ConstGenerator<OutType = P> + Neg<Output = P> + One + Neg + Clone + Copy,
{
    type Output = ArrayfireTensor<P, D>;

    fn neg(self) -> Self::Output {
        TensorOpsNeg::neg(&self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{backend::arrayfire::device::Device, Data, TensorBase};

    #[test]
    fn should_support_neg_ops() {
        let data = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = ArrayfireTensor::<f64, 2>::from_data(data, Device::CPU);

        let data_actual = tensor.neg().into_data();

        let data_expected = Data::from([[-0.0, -1.0, -2.0], [-3.0, -4.0, -5.0]]);
        assert_eq!(data_expected, data_actual);
    }
}
