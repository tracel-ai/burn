use crate::{element::TchElement, TchBackend, TchDevice};
use burn_tensor::{ops::TensorOps, Data, Shape};

lazy_static::lazy_static! {
    static ref NO_GRAD: tch::NoGradGuard = {
        tch::no_grad_guard()
    };
}

#[derive(Debug, PartialEq)]
pub struct TchTensor<P: tch::kind::Element, const D: usize> {
    pub kind: TchKind<P>,
    pub tensor: tch::Tensor,
}

impl<E: TchElement, const D: usize> std::ops::Add for TchTensor<E, D> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        TchBackend::add(&self, &rhs)
    }
}

impl<E: tch::kind::Element, const D: usize> TchTensor<E, D> {
    pub(crate) fn shape(&self) -> Shape<D> {
        Shape::from(self.tensor.size())
    }
}

unsafe impl<P: tch::kind::Element, const D: usize> Send for TchTensor<P, D> {}
unsafe impl<P: tch::kind::Element, const D: usize> Sync for TchTensor<P, D> {}

impl<P: tch::kind::Element, const D: usize> Clone for TchTensor<P, D> {
    fn clone(&self) -> Self {
        Self {
            kind: self.kind.clone(),
            tensor: self.tensor.shallow_clone(),
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct TchKind<P: tch::kind::Element> {
    _p: P,
}

impl<P: tch::kind::Element + Default> TchKind<P> {
    pub fn new() -> Self {
        Self { _p: P::default() }
    }
    pub fn kind(&self) -> tch::Kind {
        P::KIND
    }
}

impl<P: tch::kind::Element + Default, const D: usize> TchTensor<P, D> {
    pub fn from_data(data: Data<P, D>, device: tch::Device) -> Self {
        let tensor = tch::Tensor::of_slice(data.value.as_slice()).to(device);
        let shape_tch = TchShape::from(data.shape);
        let kind = TchKind::new();
        let tensor = tensor.reshape(&shape_tch.dims).to_kind(kind.kind());

        lazy_static::initialize(&NO_GRAD);
        let tensor = tensor.set_requires_grad(false);

        Self { kind, tensor }
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

impl<P: tch::kind::Element + Default + Copy + std::fmt::Debug, const D: usize> TchTensor<P, D> {
    pub fn empty(shape: Shape<D>, device: TchDevice) -> Self {
        let shape_tch = TchShape::from(shape);
        let kind = TchKind::new();
        let tensor = tch::Tensor::empty(&shape_tch.dims, (kind.kind(), device.into()));

        lazy_static::initialize(&NO_GRAD);
        let tensor = tensor.set_requires_grad(false);

        Self { kind, tensor }
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
