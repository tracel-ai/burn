use crate::tensor::{ops::TensorOpsUtilities, Data, Element, Shape, Tensor};

#[derive(Debug, PartialEq)]
pub struct TchTensor<P: tch::kind::Element, const D: usize> {
    pub kind: TchKind<P>,
    pub tensor: tch::Tensor,
    pub shape: Shape<D>,
}

unsafe impl<P: tch::kind::Element, const D: usize> Send for TchTensor<P, D> {}
unsafe impl<P: tch::kind::Element, const D: usize> Sync for TchTensor<P, D> {}

impl<P: tch::kind::Element, const D: usize> Clone for TchTensor<P, D> {
    fn clone(&self) -> Self {
        Self {
            kind: self.kind.clone(),
            tensor: self.tensor.shallow_clone(),
            shape: self.shape.clone(),
        }
    }
}

pub struct TchShape<const D: usize> {
    pub dims: [i64; D],
}

impl<const D: usize> From<Shape<D>> for TchShape<D> {
    fn from(shape: Shape<D>) -> Self {
        let mut dims = [0; D];
        for i in 0..D {
            dims[i] = shape.dims[i] as i64;
        }
        TchShape { dims }
    }
}

impl<const D: usize> From<Vec<i64>> for Shape<D> {
    fn from(shape: Vec<i64>) -> Self {
        let mut dims = [0; D];
        for i in 0..D {
            dims[i] = *shape.get(i).unwrap() as usize;
        }
        Self::new(dims)
    }
}

#[derive(Clone, Debug, PartialEq)]
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
        let shape = data.shape.clone();
        let shape_tch = TchShape::from(data.shape);
        let kind = TchKind::new();
        let tensor = tensor.reshape(&shape_tch.dims).to_kind(kind.kind());
        let tensor = tensor.set_requires_grad(false);

        Self {
            kind,
            tensor,
            shape,
        }
    }
}

impl<P: tch::kind::Element + Default + Copy + std::fmt::Debug, const D: usize> TchTensor<P, D> {
    pub fn empty(shape: Shape<D>) -> Self {
        let shape_tch = TchShape::from(shape.clone());
        let device = tch::Device::Cpu;
        let kind = TchKind::new();
        let tensor = tch::Tensor::empty(&shape_tch.dims, (kind.kind(), device.clone()));
        let tensor = tensor.set_requires_grad(false);

        Self {
            kind,
            tensor,
            shape,
        }
    }
}

impl<P: tch::kind::Element + Default + Copy + std::fmt::Debug, const D: usize>
    TensorOpsUtilities<P, D> for TchTensor<P, D>
{
    fn shape(&self) -> &Shape<D> {
        &self.shape
    }
    fn into_data(self) -> Data<P, D> {
        let values = self.tensor.into();
        Data::new(values, self.shape)
    }
    fn to_data(&self) -> Data<P, D> {
        let values = self.tensor.shallow_clone().into();
        Data::new(values, self.shape.clone())
    }
}

impl<P: Element + Into<f64> + tch::kind::Element, const D: usize> Tensor<P, D> for TchTensor<P, D> {}

#[cfg(test)]
mod tests {
    use crate::tensor::Distribution;

    use super::*;

    #[test]
    fn should_support_into_and_from_data_1d() {
        let data_expected = Data::<f32, 1>::random(Shape::new([3]), Distribution::Standard);
        let tensor = TchTensor::from_data(data_expected.clone(), tch::Device::Cpu);

        let data_actual = tensor.into_data();

        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_into_and_from_data_2d() {
        let data_expected = Data::<f32, 2>::random(Shape::new([2, 3]), Distribution::Standard);
        let tensor = TchTensor::from_data(data_expected.clone(), tch::Device::Cpu);

        let data_actual = tensor.into_data();

        assert_eq!(data_expected, data_actual);
    }
}
