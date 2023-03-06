use crate::{element::TchElement, TchBackend, TchDevice};
use burn_tensor::{ops::TensorOps, Data, Shape};
use std::{marker::PhantomData, sync::Arc};

#[derive(Debug, PartialEq)]
pub struct TchTensor<E: tch::kind::Element, const D: usize> {
    pub tensor: Arc<tch::Tensor>,
    phantom: PhantomData<E>,
}

impl<E: tch::kind::Element, const D: usize> TchTensor<E, D> {
    pub fn new(tensor: tch::Tensor) -> Self {
        Self {
            tensor: Arc::new(tensor),
            phantom: PhantomData::default(),
        }
    }
}

impl<E: TchElement, const D: usize> std::ops::Add for TchTensor<E, D> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        TchBackend::add(self, rhs)
    }
}

impl<E: tch::kind::Element, const D: usize> TchTensor<E, D> {
    pub(crate) fn shape(&self) -> Shape<D> {
        Shape::from(self.tensor.size())
    }
}

// This is safe since we don't use autodiff from LibTorch.
// Also, atommic reference counting is used to know if the tensor's data can be reused.
// If there are multiple reference on the same tensor, it becomes read only.
unsafe impl<E: tch::kind::Element, const D: usize> Send for TchTensor<E, D> {}
unsafe impl<E: tch::kind::Element, const D: usize> Sync for TchTensor<E, D> {}

impl<P: tch::kind::Element, const D: usize> TchTensor<P, D> {
    // Execute an operation on a tensor if the data can be reused.
    pub fn mut_ops<F: Fn(&mut tch::Tensor) -> O, O>(&mut self, func: F) -> Option<O> {
        let output = match Arc::get_mut(&mut self.tensor) {
            Some(tensor) => func(tensor),
            None => return None,
        };

        Some(output)
    }
    /// Execute a unary ops reusing the tensor data if possible.
    pub fn unary_ops<FOwn, FRef, O>(self, fown: FOwn, fref: FRef) -> O
    where
        FOwn: Fn(tch::Tensor) -> O,
        FRef: Fn(&tch::Tensor) -> O,
    {
        match Arc::try_unwrap(self.tensor) {
            Ok(tensor) => fown(tensor),
            Err(tensor) => fref(tensor.as_ref()),
        }
    }

    /// Execute a binary ops reusing the tensor data if possible.
    pub fn binary_ops_tensor<FLMut, FRMut, FRef, O>(
        mut lhs: Self,
        mut rhs: Self,
        flmut: FLMut,
        frmut: FRMut,
        fref: FRef,
    ) -> O
    where
        FLMut: Fn(&mut tch::Tensor, &tch::Tensor) -> O,
        FRMut: Fn(&tch::Tensor, &mut tch::Tensor) -> O,
        FRef: Fn(&tch::Tensor, &tch::Tensor) -> O,
    {
        let lhs_num_elems = lhs.shape().num_elements();
        let rhs_num_elems = rhs.shape().num_elements();

        let safe_mut_lhs = lhs_num_elems > rhs_num_elems;
        let safe_mut_rhs = rhs_num_elems > lhs_num_elems;

        if safe_mut_lhs {
            if let Some(output) = lhs.mut_ops(|lhs| flmut(lhs, &rhs.tensor)) {
                return output;
            }
        }

        if safe_mut_rhs {
            if let Some(output) = rhs.mut_ops(|rhs| frmut(&lhs.tensor, rhs)) {
                return output;
            }
        }

        fref(&lhs.tensor, &rhs.tensor)
    }
}

impl<P: tch::kind::Element, const D: usize> Clone for TchTensor<P, D> {
    fn clone(&self) -> Self {
        Self {
            tensor: self.tensor.clone(),
            phantom: PhantomData::default(),
        }
    }
}

pub struct TchShape<const D: usize> {
    pub dims: [i64; D],
}

impl<const D: usize> From<Shape<D>> for TchShape<D> {
    fn from(shape: Shape<D>) -> Self {
        let mut dims = [0; D];
        for (i, dim) in dims.iter_mut().enumerate().take(D) {
            *dim = shape.dims[i] as i64;
        }
        TchShape { dims }
    }
}

impl<E: tch::kind::Element + Default, const D: usize> TchTensor<E, D> {
    pub fn from_data(data: Data<E, D>, device: tch::Device) -> Self {
        let tensor = tch::Tensor::of_slice(data.value.as_slice()).to(device);
        let shape_tch = TchShape::from(data.shape);
        let tensor = tensor.reshape(&shape_tch.dims).to_kind(E::KIND);

        Self::new(tensor)
    }
}

#[cfg(test)]
mod utils {
    use super::*;
    use crate::{backend::TchBackend, element::TchElement};

    impl<P: TchElement, const D: usize> TchTensor<P, D> {
        pub(crate) fn into_data(self) -> Data<P, D>
        where
            P: tch::kind::Element,
        {
            <TchBackend<P> as TensorOps<TchBackend<P>>>::into_data(self)
        }
    }
}

impl<E: tch::kind::Element + Default + Copy + std::fmt::Debug, const D: usize> TchTensor<E, D> {
    pub fn empty(shape: Shape<D>, device: TchDevice) -> Self {
        let shape_tch = TchShape::from(shape);
        let tensor = tch::Tensor::empty(&shape_tch.dims, (E::KIND, device.into()));

        Self::new(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_tensor::Distribution;
    use rand::prelude::StdRng;
    use rand::SeedableRng;

    #[test]
    fn should_support_into_and_from_data_1d() {
        let data_expected = Data::<f32, 1>::random(
            Shape::new([3]),
            Distribution::Standard,
            &mut StdRng::from_entropy(),
        );
        let tensor = TchTensor::from_data(data_expected.clone(), tch::Device::Cpu);

        let data_actual = tensor.into_data();

        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_into_and_from_data_2d() {
        let data_expected = Data::<f32, 2>::random(
            Shape::new([2, 3]),
            Distribution::Standard,
            &mut StdRng::from_entropy(),
        );
        let tensor = TchTensor::from_data(data_expected.clone(), tch::Device::Cpu);

        let data_actual = tensor.into_data();

        assert_eq!(data_expected, data_actual);
    }
}
